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
static double*** tri_delta = new double**[MAXWORDS];

static VocabIndex** all_word_index = new VocabIndex*[MAXWORDS];
static VocabIndex** back_track_index = new VocabIndex*[MAXWORDS];

static VocabIndex*** tri_back_track_index = new VocabIndex**[MAXWORDS];

static int all_size[MAXWORDS] = {0};
static int viterbi_init = false;

VocabIndex getWordIndex(const Ngram& lm, VocabString word)
{
  VocabIndex word_id = lm.vocab.getIndex(word);
  if (word_id == Vocab_None)
	{
    word_id = lm.vocab.getIndex(Vocab_Unknown);
	}
  return word_id;
}

double get_lm_prob(VocabIndex word, VocabIndex context[], Ngram& lm, double unk_prob = -9999)
{
  double prob = lm.wordProb(word, context);
  if (prob == LogP_Zero)
  {
    return -999;
  }
	if (prob <= unk_prob)
	{
		return -9999;
	}
  return prob;
}


void init_Viterbi()
{
  if (viterbi_init == false)
	{
    for (int i = 0; i < MAXWORDS; i++)
		{
      delta[i] = new double[MAXMAPLEN];
      tri_delta[i] = new double*[MAXMAPLEN];

      all_word_index[i] = new VocabIndex[MAXMAPLEN];

      back_track_index[i] = new VocabIndex[MAXMAPLEN];
      tri_back_track_index[i] = new VocabIndex*[MAXMAPLEN];

      for (int j = 0; j < MAXMAPLEN; j++)
			{
        tri_delta[i][j] = new double[MAXMAPLEN];
        tri_back_track_index[i][j] = new VocabIndex[MAXMAPLEN];
      }
    }
    viterbi_init = true;
  }
  else
	{
    for(int i = 0; i < MAXWORDS; i++)
		{
      all_size[i] = 0;
    }
  }
}

void Viterbi_Algorithm(LoadMap& map, Ngram& lm, VocabString* text, int word_num)
{
  Vocab& voc_zhuyin = map.voc_zhuyin;
  Vocab& voc = map.voc_big5;
  VocabIndex context[] = {Vocab_None};
  VocabIndex word_unk_id_lm = getWordIndex(lm, Vocab_Unknown);
  double trigram_prob, unk_prob;

  init_Viterbi();

	// t0
  VocabIndex word_0_id;
  LoadMapIter word_t0_iter(map, voc_zhuyin.getIndex(text[0]));
  for(int i = 0; word_t0_iter.next(word_0_id); i++)
	{
    all_size[0] += 1;
    all_word_index[0][i] = word_0_id;
    VocabString word_t = voc.getWord(word_0_id);
    VocabIndex word_t_id = getWordIndex(lm, word_t);
		double word_id_t_prob = get_lm_prob(word_t_id, context, lm);

    delta[0][i] = word_id_t_prob;
    back_track_index[0][i] = 0;
    for(int j = 0; j < MAXMAPLEN; j++)
		{
      tri_delta[0][i][j] = word_id_t_prob;
      tri_back_track_index[0][i][j] = 0;
    }
  }

  // t1
  VocabIndex word_1_id;
  LoadMapIter word_id_t1_iter(map, voc_zhuyin.getIndex(text[1]));
  for(int i = 0; word_id_t1_iter.next(word_1_id); i++)
	{
    all_size[1] += 1;
    all_word_index[1][i] = word_1_id;
    VocabString word_1 = voc.getWord(word_1_id);
    VocabIndex word_1_id_lm = getWordIndex(lm, word_1);

    for(int j = 0; j < all_size[0]; j++)
		{
      VocabString word_1_1 = voc.getWord(all_word_index[0][j]);
      VocabIndex word_1_1_id_lm = getWordIndex(lm, word_1_1);

      VocabIndex context[] = {word_1_1_id_lm, Vocab_None};
			unk_prob = get_lm_prob(word_unk_id_lm, context, lm);

			double bigram_prob_tmp = get_lm_prob(word_1_id_lm, context, lm, unk_prob);

      double bigram_prob = bigram_prob_tmp + delta[0][j];

      trigram_prob = bigram_prob;
      tri_back_track_index[1][i][j] = j;
      tri_delta[1][i][j] = trigram_prob;
    }
  }

  //t2-tT
  for(int t = 2; t < word_num; t++)
	{
    VocabIndex word_t_id;
    LoadMapIter word_t_iter(map, voc_zhuyin.getIndex(text[t]));
    for(int i = 0; word_t_iter.next(word_t_id); i++)
		{
      all_size[t] += 1;
      all_word_index[t][i] = word_t_id;
      for(int j = 0; j < all_size[t-1]; j++)
			{
        tri_delta[t][i][j] = -9999;
      }
    }

    for(int i = 0; i < all_size[t-1]; i++)
		{
      for(int j = 0; j < all_size[t-2]; j++)
			{

        if(tri_delta[t-1][i][j] > -9999 )
				{

          VocabString word_t1, word_t2;
          VocabIndex word_t1_id_lm, word_t2_id_lm;

          word_t1 = voc.getWord(all_word_index[t-1][i]);
          word_t1_id_lm = getWordIndex(lm, word_t1);

          word_t2 = voc.getWord(all_word_index[t-2][j]);
          word_t2_id_lm = getWordIndex(lm, word_t2);

          VocabIndex context[] = {word_t1_id_lm, word_t2_id_lm, Vocab_None};

          for(int k = 0; k < all_size[t]; k++)
					{
            VocabString word_t = voc.getWord(all_word_index[t][k]);
            VocabIndex word_t_id_lm = getWordIndex(lm, word_t);

						unk_prob = get_lm_prob(word_unk_id_lm, context, lm);
						trigram_prob = get_lm_prob(word_t_id_lm, context, lm, unk_prob);

            if(trigram_prob > -9999)
						{
              trigram_prob += tri_delta[t-1][i][j];
              if(trigram_prob > tri_delta[t][k][i])
							{
                tri_delta[t][k][i] = trigram_prob;
                tri_back_track_index[t][k][i] = j;
              }
            }
          }
        }
      }
    }
  }

  double max_path_prob = -1e10;
  int max_path_idxi = -1;
  int max_path_idxj = -1;
  for(int i = 0; i < all_size[word_num-1]; i++)
	{
    for(int j = 0; j < all_size[word_num-2]; j++)
		{
      double cur_prob = tri_delta[word_num-1][i][j];
      if (cur_prob > max_path_prob)
			{
        max_path_prob = cur_prob;
        max_path_idxi = i;
        max_path_idxj = j;
      }
    }
  }

  int path_index[MAXWORDS];
  path_index[word_num - 1] = max_path_idxi;
  path_index[word_num - 2] = max_path_idxj;
  for(int t = word_num-3; t >= 0; t--)
	{
    path_index[t] = tri_back_track_index[t+2][path_index[t+2]][path_index[t+1]];
  }
  for(int t = 0; t < word_num; t++)
	{
    text[t] = voc.getWord(all_word_index[t][path_index[t]]);
  }
}

int main(int argc, char *argv[])
{
  char* segmented_file_path = argv[1];
  char* zhuyin_big5_mapping_path = argv[2];
  char* language_model_path = argv[3];
  char* output_file_path = argv[4];

  Vocab voc_lm, voc_zhuyin, voc;

  Ngram lm(voc_lm, 3);
  {
    File lm_file(language_model_path, "r");
    lm.read(lm_file);
    lm_file.close();
  }

  LoadMap zhuyin_big5_map;
  zhuyin_big5_map.read(zhuyin_big5_mapping_path);

  File segmented_data(segmented_file_path, "r");
  ofstream output_file(output_file_path);

  char* line = NULL;
  while(line = segmented_data.getline())
	{
    VocabString text[MAXWORDS];
    int word_num = Vocab::parseWords(line, &text[1], MAXWORDS);
    text[0] = Vocab_SentStart;
    text[word_num+1] = Vocab_SentEnd;

    Viterbi_Algorithm(zhuyin_big5_map, lm, text, word_num+2);

    for(int i = 0; i < word_num + 1; i++)
		{
      output_file << text[i] << " ";
    }
    output_file << text[word_num+1] << endl;
  }
  segmented_data.close();
  output_file.close();
}
