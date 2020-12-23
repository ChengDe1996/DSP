#include <stdio.h>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <limits>

#include "Ngram.h"
#include "Vocab.h"
#include "File.h"

#ifndef MAXWORDS
#define MAXWORDS 256
#endif

#ifndef MAXMAPLEN
#define MAXMAPLEN 1024
#endif

using namespace std;

class LoadMap
{
  friend class LoadMapIter;
public:
  map<VocabIndex, set<VocabIndex>> zhuyin2big5_map;
  Vocab voc_zhuyin, voc_big5;
  void read(char* file);
  void write(char* file);
};

void LoadMap::read(char* file)
{
  File input_file(file, "r");
  char* line = NULL;

  VocabIndex zhuyin_sent_start = voc_zhuyin.addWord(Vocab_SentStart);
  zhuyin2big5_map[zhuyin_sent_start].insert(voc_big5.addWord(Vocab_SentStart));

  VocabIndex zhuyin_sent_end = voc_zhuyin.addWord(Vocab_SentEnd);
  zhuyin2big5_map[zhuyin_sent_end].insert(voc_big5.addWord(Vocab_SentEnd));

  while(line = input_file.getline())
  {
    VocabString text[2048];
    int word_num = Vocab::parseWords(line, text, 2048);
    VocabIndex zhuyin_index = voc_zhuyin.addWord(text[0]);
    for (int i = 1; i < word_num; i++)
    {
      VocabIndex big5_index = voc_big5.addWord(text[i]);
      zhuyin2big5_map[zhuyin_index].insert(big5_index);
    }
  }
}

class LoadMapIter
{
public:
  LoadMapIter(LoadMap &map, VocabIndex vocab_index);
  Boolean next(VocabIndex &w);
  VocabIndex vocab_index;
  set<VocabIndex>* vocab_set;
  set<VocabIndex>::iterator vocab_iter;
};

LoadMapIter::LoadMapIter(LoadMap &map, VocabIndex vocab_index_in)
{
  vocab_index = vocab_index_in;
  vocab_set = &(map.zhuyin2big5_map[vocab_index]);
  vocab_iter = vocab_set->begin();
}

Boolean LoadMapIter::next(VocabIndex &w)
{
  if(vocab_iter != vocab_set->end())
  {
    w = *vocab_iter;
    vocab_iter++;
    return true;
  }
  return false;
}

static double** delta = new double*[MAXWORDS];
static VocabIndex** all_word_index = new VocabIndex*[MAXWORDS];
static VocabIndex** back_track_index = new VocabIndex*[MAXWORDS];
static int all_size[MAXWORDS] = {0};
static bool viterbi_init = false;

VocabIndex getWordIndex(const Ngram& lm, VocabString word)
{
  VocabIndex word_id = lm.vocab.getIndex(word);
  if (word_id == Vocab_None)
  {
    word_id = lm.vocab.getIndex(Vocab_Unknown);
  }
  return word_id;
}

double get_lm_prob(VocabIndex word, VocabIndex context[], Ngram& lm)
{
  double prob = lm.wordProb(word, context);
  if (prob == LogP_Zero)
  {
    return -999;
  }
  return prob;
}

double bigram_score(VocabString word_predict, VocabString word_given, Ngram &lm)
{
  VocabIndex word_predict_id = getWordIndex(lm, word_predict);
  VocabIndex word_given_id = getWordIndex(lm, word_given);
  VocabIndex context[] = { word_given_id, Vocab_None};
  return lm.wordProb(word_predict_id, context);
}

void Viterbi_initialization()
{
  if (viterbi_init == false)
  {
    for (int i = 0; i < MAXWORDS; i++)
    {
      delta[i] = new double[MAXMAPLEN];
      all_word_index[i] = new VocabIndex[MAXMAPLEN];
      back_track_index[i] = new VocabIndex[MAXMAPLEN];
    }
    viterbi_init = true;
  }
  for(int i = 0; i < MAXWORDS; i++)
  {
    all_size[i] = 0;
  }
}

void Viterbi_Algorithm(LoadMap& map, Ngram& lm, VocabString* text, int word_num)
{
  Vocab& voc_zhuyin = map.voc_zhuyin;
  Vocab& voc_c = map.voc_big5;
  VocabIndex contextNone[] = {Vocab_None};

  Viterbi_initialization();

  VocabIndex word_0_id;
  LoadMapIter word_t0_iter(map, voc_zhuyin.getIndex(text[0]));

  //t0
  for(int i = 0; word_t0_iter.next(word_0_id); i++)
  {
    VocabString word_t_word = voc_c.getWord(word_0_id);
    VocabIndex word_t_id = getWordIndex(lm, word_t_word);
    double word_t_prob = get_lm_prob(word_t_id, contextNone, lm);
    delta[0][i] = word_t_prob;
    all_word_index[0][i] = word_0_id;
    back_track_index[0][i] = 0;
    all_size[0] += 1;
  }

  //t1-tT
  for(int t = 1; t < word_num; t++)
  {
    VocabIndex word_t_id;
    LoadMapIter word_t_iter(map, voc_zhuyin.getIndex(text[t]));

    for(int i = 0; word_t_iter.next(word_t_id); i++)
    {
      VocabString word_t = voc_c.getWord(word_t_id);
      double max_prob = -1e10;

      for(int j = 0; j < all_size[t-1]; j++)
      {
        VocabString word_t1 = voc_c.getWord(all_word_index[t-1][j]);
        double bigram_prob = bigram_score(word_t, word_t1, lm);

        bigram_prob += delta[t-1][j];
        if (bigram_prob > max_prob)
        {
          max_prob = bigram_prob;
          back_track_index[t][i] = j;
        }
      }

      delta[t][i] = max_prob;
      all_word_index[t][i] = word_t_id;
      all_size[t] += 1;
    }
  }

  double max_path_prob = -1e10;
  int max_path_index = -1;
  
  for(int i = 0; i < all_size[word_num-1]; i++)
  {
    double temp_prob = delta[word_num-1][i];
    if (temp_prob > max_path_prob)
    {
      max_path_prob = temp_prob;
      max_path_index = i;
    }
  }

  int path_index[MAXWORDS];
  path_index[word_num - 1] = max_path_index;

  for(int t = word_num-2; t >= 0; t--)
  {
    path_index[t] = back_track_index[t+1][path_index[t+1]];
  }

  for(int t = 0; t < word_num; t++)
  {
    text[t] = voc_c.getWord(all_word_index[t][path_index[t]]);
  }
}

int main(int argc, char *argv[])
{
    char* segmented_file_path = argv[1];
    char* zhuyin2big5_map_path = argv[2];
    char* language_model_path = argv[3];
    char* output_file_path = argv[4];

    Vocab vocab_lm, vocab_zhuyin, vocab;

    Ngram lm(vocab_lm, 2);
    {
      File lm_file(language_model_path, "r");
      lm.read(lm_file);
      lm_file.close();
    }

    LoadMap zhuyin2big5_map;
    zhuyin2big5_map.read(zhuyin2big5_map_path);

    File segmented_file(segmented_file_path, "r");
    ofstream output_file(output_file_path);

    char* line = NULL;
    while(line = segmented_file.getline())
    {
      VocabString text[MAXWORDS];
      int word_num = Vocab::parseWords(line, &text[1], MAXWORDS);
      text[0] = Vocab_SentStart;
      text[word_num+1] = Vocab_SentEnd;

      Viterbi_Algorithm(zhuyin2big5_map, lm, text, word_num+2);

      for(int i = 0; i < word_num + 1; i++)
      {
        output_file << text[i] << " ";
      }
      output_file << text[word_num+1] << endl;
    }

    segmented_file.close();
    output_file.close();
}
