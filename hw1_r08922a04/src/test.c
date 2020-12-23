#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmm.h"

double viterbi(HMM *hmm, int seq_to_int[MAX_SEQ], char* sequence)
{
    double delta[MAX_SEQ][MAX_STATE] = {{0.0}};
    
    for(int i = 0; i < hmm -> state_num; i ++)
    {
        delta[0][i] = hmm -> initial[i] * hmm -> observation[seq_to_int[0]][i];
    }
    int length = 0;
    length = strlen(sequence);
    
    for(int t = 0; t < length - 1; t ++)
    {
        for(int j = 0; j < hmm -> state_num; j ++)
        {
            double max_value = 0.0;
            for(int i = 0; i < hmm -> state_num; i ++)
            {
                double temp = 0;
                temp = delta[t][i] * (hmm -> transition[i][j]);
                if(temp > max_value)
                {
                    max_value = temp;
                }
            }
            delta[t+1][j] = max_value * (hmm -> observation[seq_to_int[t+1]][j]);
        }
    }

    double max_prob = 0.0;

    for(int i = 0; i < hmm -> state_num; i ++)
    {
        if(delta[length - 1][i] > max_prob)
        {
            max_prob = delta[length - 1][i];
        }
    }
    return max_prob;
}


int main(int argc, char const *argv[])
{
    const char* model_list_path = argv[1];
    const char* seq_path = argv[2];
    const char* output_result_path = argv[3];

    HMM hmm[5];
    load_models(model_list_path, hmm, 5);
    
    FILE *data = open_or_die(seq_path, "r");
    FILE *result = open_or_die(output_result_path, "w");
    
    char sequence[MAX_SEQ] = {'\0'};
    int seq_to_int[MAX_SEQ] = {0};

    while(fscanf(data, "%s", sequence) != EOF)
    {
        int length = 0;
        length = strlen(sequence);
        for(int i = 0; i < length ; i ++)
        {
            seq_to_int[i] = sequence[i] - 'A';
        }
        double max_prob = 0.0;
        int max_model = -1; 
        for(int i = 0; i < 5; i++)
        {
            double temp_prob = 0.0;
            temp_prob = viterbi(&hmm[i], seq_to_int, sequence);
            if(temp_prob > max_prob)
            {
                max_prob = temp_prob;
                max_model = i;
            }
        }
        fprintf(result, "%s %e\n", hmm[max_model].model_name, max_prob);
    }
    fclose(data);
    fclose(result);
    return 0;
}