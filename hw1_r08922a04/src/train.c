#include "hmm.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void forward_algorithm(HMM *hmm, double alpha[MAX_SEQ][MAX_STATE], int seq_to_int[MAX_SEQ], int seq_len)
{
    for(int i = 0; i < hmm -> state_num; i++)
    {
        alpha[0][i] = hmm -> initial[i] * hmm -> observation[seq_to_int[0]][i];
    }
    for(int t = 1; t < seq_len; t ++)
    {
        for(int j = 0; j < hmm -> state_num; j ++)
        {
            for(int i = 0; i < hmm -> state_num; i ++)
            {
                alpha[t][j] += alpha[t-1][i] * (hmm -> transition[i][j]) * hmm -> observation[seq_to_int[t]][j];
            }
        }
    }
    return;
}

void backward_algorithm(HMM *hmm, double beta[MAX_SEQ][MAX_STATE], int seq_to_int[MAX_SEQ], int seq_len)
{
    for(int i = 0; i < hmm -> state_num; i ++)
    {
        beta[seq_len - 1][i] = 1.0;
    }
    for(int t = seq_len - 2; t >= 0; t --)
    {
        for(int i = 0; i < hmm -> state_num; i ++)
        {
            for(int j = 0; j < hmm -> state_num; j ++)
            {
                beta[t][i] += hmm -> transition[i][j] * hmm -> observation[seq_to_int[t+1]][j] * beta[t+1][j];
            }
        }
    }
    return;
}

void forward_backward(HMM *hmm, double alpha[MAX_SEQ][MAX_STATE], double beta[MAX_SEQ][MAX_STATE], 
                        double gamma[MAX_SEQ][MAX_STATE], double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE], 
                        int seq_to_int[MAX_SEQ], int seq_len)
{
    for(int t = 0; t < seq_len; t ++)
    {
        double gamma_sum = 0.0;
        for(int i = 0; i < hmm -> state_num; i ++)
        {
            gamma[t][i] = alpha[t][i] * beta[t][i];
            gamma_sum += gamma[t][i];
        }
        for(int i = 0; i < hmm -> state_num; i ++)
        {
            gamma[t][i] /= gamma_sum;
        }
    }

    for(int t = 0; t < seq_len; t ++)
    {
        double epsilon_sum = 0.0;
        for(int i = 0; i < hmm -> state_num; i ++)
        {
            for(int j = 0; j < hmm -> state_num; j ++)
            {
                epsilon[t][i][j] = alpha[t][i] * hmm -> transition[i][j] * hmm -> observation[seq_to_int[t+1]][j] * beta[t+1][j];
                epsilon_sum += epsilon[t][i][j];
            }
        }
        for(int i = 0; i < hmm -> state_num; i ++)
        {
            for(int j = 0; j < hmm -> state_num; j ++)
            {
                epsilon[t][i][j] /= epsilon_sum;
            }
        }
    }
    return;
}

void accumulation(HMM *hmm, double gamma[MAX_SEQ][MAX_STATE], double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE], 
                    double pi[MAX_STATE], int seq_to_int[MAX_SEQ], int seq_len, 
                    double epsilon_sum[MAX_STATE][MAX_STATE], double gamma_sum_a[MAX_STATE][MAX_STATE], 
                    double gamma_observation[MAX_OBSERV][MAX_STATE], double gamma_sum_b[MAX_OBSERV][MAX_STATE])
{
    for(int i = 0; i < hmm -> state_num; i ++)
    {
        pi[i] += gamma[0][i];
    }

    for(int i = 0; i < hmm -> state_num; i ++)
    {
        for(int j = 0; j < hmm -> state_num; j ++)
        {
            for(int t = 0; t < (seq_len - 1); t ++)
            {
                epsilon_sum[i][j] += epsilon[t][i][j];
                gamma_sum_a[i][j] += gamma[t][i]; // 1 ~ T-1
            }
        }
    }

    for(int i = 0; i < hmm -> observ_num; i ++)
    {
        for(int j = 0; j < hmm -> state_num; j ++)
        {
            for(int t = 0; t < seq_len; t ++)
            {
                if(seq_to_int[t] == i)
                {
                    gamma_observation[i][j] += gamma[t][j];
                }
                gamma_sum_b[i][j] += gamma[t][j]; // 1 ~ T
            }
        }
    }
    return;
}

void update(HMM *hmm, double pi[MAX_STATE], int data_len, 
            double epsilon_sum[MAX_STATE][MAX_STATE], 
            double gamma_sum_a[MAX_STATE][MAX_STATE], 
            double gamma_observation[MAX_OBSERV][MAX_STATE], 
            double gamma_sum_b[MAX_OBSERV][MAX_STATE])
{
    for(int i = 0; i < hmm -> state_num; i ++)
    {
        hmm -> initial[i] = pi[i] / data_len;
    }

    for(int i = 0; i < hmm -> state_num; i ++)
    {
        for(int j = 0; j < hmm -> state_num; j ++)
        {
            hmm -> transition[i][j] = epsilon_sum[i][j] / gamma_sum_a[i][j];
        }
    }

    for(int i = 0; i < hmm -> observ_num; i ++)
    {
        for(int j = 0; j < hmm -> state_num; j ++)
        {
            hmm -> observation[i][j] = gamma_observation[i][j] / gamma_sum_b[i][j];
        }
    }
    return;
}

void train(HMM *hmm, const char *train_file){
    FILE *train_data = open_or_die(train_file, "r");
    char sequence[MAX_SEQ] = {'\0'};

    double pi[MAX_STATE] = {0.0};
    int seq_to_int[MAX_SEQ] = {0};
    int seq_len = 0;
    int data_len = 0;

    double epsilon_sum[MAX_STATE][MAX_STATE] = {{0.0}};
    double gamma_sum_a[MAX_STATE][MAX_STATE] = {{0.0}};
    double gamma_observation[MAX_OBSERV][MAX_STATE] = {{0.0}}; 
    double gamma_sum_b[MAX_OBSERV][MAX_STATE] = {{0.0}};

    while(fscanf(train_data, "%s", sequence) != EOF){
        data_len ++;
        seq_len = strlen(sequence);
        for(int i = 0; i < seq_len ; i ++)
        {
            seq_to_int[i] = sequence[i] - 'A';
        }

        double alpha[MAX_SEQ][MAX_STATE] = {{0.0}};
        double beta[MAX_SEQ][MAX_STATE] = {{0.0}};
        forward_algorithm(hmm, alpha, seq_to_int, seq_len);
        backward_algorithm(hmm, beta, seq_to_int, seq_len);

        double gamma[MAX_SEQ][MAX_STATE] = {{0.0}};
        double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE] = {{{0.0}}};
        forward_backward(hmm, alpha, beta, gamma, epsilon, seq_to_int, seq_len);
        accumulation(hmm, gamma, epsilon, pi, seq_to_int, seq_len, epsilon_sum, gamma_sum_a, gamma_observation, gamma_sum_b);
    }
    update(hmm, pi, data_len, epsilon_sum, gamma_sum_a, gamma_observation, gamma_sum_b);
    fclose(train_data);
    return;
}

int main(int argc, char const *argv[])
{

    int iterations = atoi(argv[1]);
    const char* model_init_path = argv[2];
    const char* seq_path = argv[3];
    const char* output_model_path = argv[4];

    HMM hmm;
    loadHMM(&hmm, model_init_path);

    for(int i = 0; i < iterations; i++)
    {
        train(&hmm, seq_path);
    }

    FILE* model = open_or_die(output_model_path, "w");

    dumpHMM(model, &hmm);
    fclose(model);
    return 0;
}