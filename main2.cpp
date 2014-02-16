#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <algorithm>

#define TIGER_NUMACTIONS 2
#define TIGER_NUMSTATES 2
#define TIGER_NUMOBS 2

#define NUM_BELIEFS 50

#include "planning.hpp"

using namespace std;

double logmul (double a, double b) {
    return a + b;
}

double logadd (double a, double b) {
    double m = max(a, b);
    if (exp(m) == 0.0)
    {
        return m;
    }
    else
    {
        return m + log(exp(a - m) + exp(b - m));
    }
}

#define TNS TIGER_NUMSTATES
#define TNA TIGER_NUMACTIONS
#define TNO TIGER_NUMOBS

//#define TIGER_REWARD 10
//#define TIGER_PENALTY (-100)

#define USE_REWARDS true

#define SMALL_REWARD 1
#define BIG_REWARD 2
#define SUCCESS_PROB 0.1
#define OBS_SUCCESS 0.8

#define START_STATE (0)
//#define START_STATE (rand() % pomdp.numstates)

#define OPT true
#define FILENAME "2sensor_opt_rewards_100alpha.txt"

// ofstream outFile;
clock_t t;
clock_t rept;

// two states, left and right
// two actions, left and right
// going left in left gets some reward
// going right in right gets some reward
// you start off in left
// it's hard to transition to the right, but there's a high reward in the right
// observations are known, and transitions need to be learned
struct POMDP
{
    int numstates;
    int numactions;
    int numobs;

    double gamma;
    double rmax;

    double t[TNS][TNA][TNS];
    double o[TNS][TNA][TNO];
    double r[TNS][TNA];

    vector<int> actions;
    vector<int> obs;
    vector<int> rewards;
    
    // useful for debugging
    vector<int> states;

    int curr_state;

    POMDP()
        : numstates(TNS), numactions(TNA), numobs(TNO),
          gamma(0.99), rmax(BIG_REWARD)
    {
        // actions are: go left or go right
        // states are: left, right
        // obs are: hear left, hear right
        // obs are fixed

        // the only rewards are going left in left and going right in right
        r[0][0] = SMALL_REWARD;
        r[0][1] = 0;

        r[1][0] = 0;
        r[1][1] = BIG_REWARD;
        
        // transitions are mostly deterministic except for trying to get to right from left
        t[0][0][0] = 1.0; // stay in the same state
        t[0][0][1] = 0.0;
        
        t[0][1][0] = 1.0 - SUCCESS_PROB; // hard to go right
        t[0][1][1] = SUCCESS_PROB;
        
        t[1][0][0] = 1.0; // easily go back to left
        t[1][0][1] = 0.0;
        
        t[1][1][0] = 0.0; // stay right
        t[1][1][1] = 1.0;
        
        // observations gives the right state most of the time
        // doesn't matter which action got you there
        o[0][0][0] = OBS_SUCCESS;
        o[0][0][1] = 1.0 - OBS_SUCCESS;
        
        o[0][1][0] = OBS_SUCCESS;
        o[0][1][1] = 1.0 - OBS_SUCCESS;
        
        o[1][0][0] = 1.0 - OBS_SUCCESS;
        o[1][0][1] = OBS_SUCCESS;
        
        o[1][1][0] = 1.0 - OBS_SUCCESS;
        o[1][1][1] = OBS_SUCCESS;

        // start
        //curr_state = rand() % numstates;
        curr_state = START_STATE;
    }
    void step(int action)
    {
        int prev_state = curr_state;
        (void)prev_state;

        // advance to the next state
        double accum = 0.0;
        double target = sample_unif();
        int next_state = -1;
        for (int i = 0; i < numstates; ++i)
        {
            accum += t[curr_state][action][i];
            if (accum >= target)
            {
                next_state = i;
                break;
            }
        }
        assert(next_state >=0);

        // sample an obs
        int new_obs = -1;
        accum = 0.0;
        target = sample_unif();
        for (int i = 0; i < numobs; ++i)
        {
            accum += o[next_state][action][i];
            if (accum >= target)
            {
                new_obs = i;
                break;
            }
        }
        // cout << new_obs << endl;
        // for (int i = 0; i < TIGER_NUMSTATES; ++i)
        // {
        //         for (int j = 0; j < TIGER_NUMACTIONS; ++j)
        //         {
        //                 for (int k = 0; k < TIGER_NUMOBS; ++k)
        //                 {
        //                         //cout << plan.opt_z[i][j][k] << " ";
        //                         cout << o[i][j][k] << " ";
        //                 }
        //                 cout << "|";
        //         }
        //         cout << endl;
        // }
        assert(new_obs >= 0);
        // update the stuff
        actions.push_back(action);
        obs.push_back(new_obs);
        rewards.push_back(r[curr_state][action]);
        states.push_back(next_state);
        curr_state = next_state;

        //cout << "action " << action << " " << prev_state << " -> " << next_state << endl;
    }
    void set_o(double (&new_o)[TNS][TNA][TNO])
    {
        for (int i = 0; i < numstates; ++i)
        {
            for (int j = 0; j < numactions; ++j)
            {
                for (int k = 0; k < numobs; ++k)
                {
                    o[i][j][k] = new_o[i][j][k];
                }
            }
        }
    }
    void set_tr(double (&new_tr)[TNS][TNA][TNS])
    {
        for (int i = 0; i < numstates; ++i)
        {
            for (int j = 0; j < numactions; ++j)
            {
                for (int k = 0; k < numstates; ++k)
                {
                    t[i][j][k] = new_tr[i][j][k];
                }
            }
        }
    }
};

// only initialize for transitions
void initialize(POMDP &pomdp, double (&tr)[TNS][TNA][TNS])
{
    // completely random initialization
    for (int i = 0; i < pomdp.numstates; ++i)
    {
        for (int j = 0; j < pomdp.numactions; ++j)
        {
            double total = 0;
            for (int k = 0; k < pomdp.numstates; ++k)
            {
                double p = sample_unif();
                //double p = 0.5;
                tr[i][j][k] = p;
                total += p;
            }
            for (int k = 0; k < pomdp.numobs; ++k)
            {
                tr[i][j][k] /= total;
            }
        }
    }
}

// state vector
struct SVec
{
    double &operator[](int i)
    {
        return arr[i];
    }
    double arr[TNS];
};

// given an initialization, output the learned obs
void em(POMDP &pomdp, double (&tr)[TNS][TNA][TNS], double (&best_tr)[TNS][TNA][TNS])
{
    double max = 1;
    const int T = pomdp.obs.size() - 1;
    vector<SVec> alpha(T+1);
    vector<SVec> beta(T+1);
    vector<SVec> gamma(T+1);
    
    // at time t, the current state is pomdp.states[t]
    // pomdp.rewards[t] is the reward for the previous state
    // pomdp.obs[t] is the current obs
    // pomdp.actions[t] is the previous action
    
    double pi [2] = {0.5, 0.5};
    for (int iters = 0; iters < 5 and max > 0.001; iters++) {
        // This should be initialized outside of the loop, but for some odd reason
        // its value seems to change after each iteration...so we'll put it here for now.
        max = 1;
        for (int i = 0; i < pomdp.numstates; i++) {
            alpha[0][i] = logmul(log(pi[i]), log(pomdp.o[i][pomdp.actions[0]][pomdp.obs[0]]));
        }
        for (int l = 1; l <= T; l++) {
            for (int j = 0; j < pomdp.numstates; j++) {
                double sum = log(0.0);
                for (int i = 0; i < pomdp.numstates; i++) {
                    sum = logadd(sum, logmul(alpha[l - 1][i], log(tr[i][pomdp.actions[l]][j])));
                }
                alpha[l][j] = logmul(sum, log(pomdp.o[j][pomdp.actions[l]][pomdp.obs[l]]));
            }
            if (USE_REWARDS and l < T)
            {
                assert(pomdp.numstates==2);
                if (abs (pomdp.rewards[l+1] - SMALL_REWARD) < 0.0001)
                {
                    // clear and set the state to be left
                    alpha[l][0] = log(1);
                    alpha[l][1] = log(0);
                }
                else if (abs (pomdp.rewards[l+1] - BIG_REWARD) < 0.0001)
                {
                    // clear and set the state to be right
                    alpha[l][0] = log(0);
                    alpha[l][1] = log(1);
                }
            }
        }

        for (int i = 0; i < pomdp.numstates; i++) {
            beta[T][i] = log(1);
        }
        for (int l = T - 1; l >= 0; l--) {
            for (int i = 0; i < pomdp.numstates; i++) {
                beta[l][i] = log(0.0);
                for (int j = 0; j < pomdp.numstates; j++) {
                    double prod = logmul(log(tr[i][pomdp.actions[l + 1]][j]), log(pomdp.o[j][pomdp.actions[l + 1]][pomdp.obs[l + 1]]));
                    prod = logmul(prod, beta[l + 1][j]);
                    beta[l][i] = logadd(beta[l][i], prod);
                }
            }
            if (USE_REWARDS and l < T)
            {
                assert(pomdp.numstates==2);
                if (abs (pomdp.rewards[l+1] - SMALL_REWARD) < 0.0001)
                {
                    // clear and set the state to be left
                    beta[l][0] = log(1);
                    beta[l][1] = log(0);
                }
                else if (abs (pomdp.rewards[l+1] - BIG_REWARD) < 0.0001)
                {
                    // clear and set the state to be right
                    beta[l][0] = log(0);
                    beta[l][1] = log(1);
                }
            }
        }
        
        // calculate gamma = alpha*beta
        for (int l = 0; l <= T; l++) {
            double sum = 0;
            for (int i = 0; i < pomdp.numstates; i++) {
                if (i == 0) {
                    sum = logmul(alpha[l][i], beta[l][i]);
                }
                else {
                    sum = logadd(sum, logmul(alpha[l][i], beta[l][i]));
                }
            }
            for (int i = 0; i < pomdp.numstates; i++) {
                gamma[l][i] = logmul(logmul(alpha[l][i], beta[l][i]), -sum);
            }
        }
        
        // now to infer transitions
        // gamma_action_sum is the expected number of times (s,a) occurred
        double gamma_action_sum[TNS][TNA];
        // the expected number of times (s,a,s') occurred
        double epsilon_sum[TNS][TNA][TNS];
        
        // time to calculate gamma_action_sum and epsilon_sum
        // initialize gamma_action_sum and epsilon_sum to all zeros
        for (int x = 0; x < pomdp.numstates; ++x)
        {
            for (int y = 0; y < pomdp.numactions; ++y)
            {
                gamma_action_sum[x][y] = log(0.0);
                for (int z = 0; z < pomdp.numstates; ++z)
                {
                    epsilon_sum[x][y][z] = log(0.0);
                }
            }
        }
        // add up according to actions up to but not including the last timestep
        for (int l = 0; l <= T-1; ++l)
        {
            // sum for normalization later
            double sum = log(0.0);
            // first calculate the values for these states
            double temp_vals[TNS][TNS];
            // current action
            int cact = pomdp.actions[l+1];
            
            for (int x = 0; x < pomdp.numstates; ++x)
            {
                
                // update only the entries corresponding to the action at this timestep
                gamma_action_sum[x][cact] = logadd(gamma[l][x],gamma_action_sum[x][cact]);
                
                for (int z = 0; z < pomdp.numstates; ++z)
                {
                    // numerator of the expression
                    // instead of calling logmul I just use +
                    double top = alpha[l][x] + tr[x][cact][z] + pomdp.o[z][cact][pomdp.obs[l+1]] + beta[l+1][z];
                    sum = logadd(sum, top);
                    temp_vals[x][z] = top;
                }
            }
            
            // next normalize and add to the sum
            for (int x = 0; x < pomdp.numstates; ++x)
            {
                for (int z = 0; z < pomdp.numstates; ++z)
                {
                    if (sum != log(0.0))
                    {
                        // subtract to divide to normalize
                        temp_vals[x][z] -= sum;
                        // acc it
                        epsilon_sum[x][cact][z] = logadd(temp_vals[x][z],epsilon_sum[x][cact][z]);
                    }
                    // otherwise don't add anything
                    
                }
            }
        }
        
        // now it's easy to compute the estimated probs using the sum variables above
        for (int x = 0; x < pomdp.numstates; ++x)
        {
            for (int y = 0; y < pomdp.numactions; ++y)
            {
                // for normalization fixing stuff up
                double sum = 0.0;
                for (int z = 0; z < pomdp.numstates; ++z)
                {
                    if (gamma_action_sum[x][y] == log(0.0))
                    {
                        // if no data, then uniform
                        tr[x][y][z] = 1.0/pomdp.numstates;
                    }
                    else
                    {
                        tr[x][y][z] = exp(epsilon_sum[x][y][z] - gamma_action_sum[x][y]);
                    }
                    sum += tr[x][y][z];
                }
                // normalize to fix stuff up
                // not sure why this isn't already normalized - they should be
                for (int z = 0; z < pomdp.numstates; ++z)
                {
                    tr[x][y][z] /= sum;
                }
            }
        }
        
        // for debugging
        if (0)
        {
            cout << "printing out gamma action sum and epsilon sum" << endl;
            for (int x = 0; x < pomdp.numstates; ++x)
            {
                for (int y = 0; y < pomdp.numactions; ++y)
                {
                    cout << x << "," << y << " | " << exp(gamma_action_sum[x][y]) << endl;
                }
            }
            cout << "---------" << endl;
            for (int x = 0; x < pomdp.numstates; ++x)
            {
                for (int y = 0; y < pomdp.numactions; ++y)
                {
                    for (int z = 0; z < pomdp.numstates; ++z)
                    {
                        cout << x << "," << y << "," << z << " | " << exp(epsilon_sum[x][y][z]) << endl;
                    }
                }
            }
        }
        // for debugging
        if (0)
        {
            cout << "estimated transitions" << endl;
            for (int x = 0; x < pomdp.numstates; ++x)
            {
                for (int y = 0; y < pomdp.numactions; ++y)
                {
                    for (int z = 0; z < pomdp.numstates; ++z)
                    {
                        cout << x << "," << y << "," << z << " | " << tr[x][y][z] << endl;
                    }
                }
            }
        }
    }
    for (int x = 0; x < pomdp.numstates; x++) {
        for (int y = 0; y < pomdp.numactions; y++) {
            for (int z = 0; z < pomdp.numstates; z++) {
                best_tr[x][y][z] = tr[x][y][z];
            }
        }
    }
    
    // for debugging
    if (0)
    {
        cout << "em called" << endl;
        for (int l = 0; l <= T; l++){
            cout << "sim step " << l << ", s " << pomdp.states[l] << ", r " << pomdp.rewards[l] << ", a " << pomdp.actions[l] << ", o " << pomdp.obs[l] << ": ";
            for (int i = 0; i < pomdp.numstates; i++) {
                cout << exp(gamma[l][i]) << " ";
            }
            cout << "| ";
            for (int i = 0; i < pomdp.numstates; i++) {
                cout << exp(alpha[l][i]) << " ";
            }
            cout << "| ";
            for (int i = 0; i < pomdp.numstates; i++) {
                cout << exp(beta[l][i]) << " ";
            }
            cout << endl;
        }
    }
    // for (int x = 0; x < pomdp.numstates; x++){
    //         for (int y = 0; y < pomdp.numactions; y++){
    //                 for (int z = 0; z < pomdp.numobs; z++){
    //                         learned_o[x][y][z] = best_o[x][y][z];
    //                         cout << learned_o[x][y][z] << endl;
    //                 }
    //         }
    // }
}

int main()
{
    srand(0);
    //srand(time(0));
    // outFile.open("outstream_100alpha.txt", ios::out | ios::app);
    cout << "hello all" << endl;

    int B = 10;
    int reps = 1;
    int steps = 100;
    double sum_rewards = 0;
    int prev_sum = 0;

    vector<double> rs(steps, 0.0);

    double zeros[TNS][TNA][TNO];
    double tr[TNS][TNA][TNS];
    double err[TNS][TNA][TNS];
    for (int x = 0; x < TNS; x++) {
        for (int y = 0; y < TNA; y++) {
            for (int z = 0; z < TNO; z++) {
                zeros[x][y][z] = 0;
            }
            for (int z = 0; z < TNS; z++) {
                tr[x][y][z] = 0;
                err[x][y][z] = 0;
            }
        }
    }

    ofstream rewardsFile;
    rewardsFile.open(FILENAME, ios::out | ios::app);

    for (int rep = 0; rep < reps; ++rep)
    {
        // rept = clock();
        // cout << rep << endl;
        POMDP pomdp;
        initialize(pomdp, tr);
        Planning<POMDP,double[TNS][TNA][TNS],double[TNS][TNA][TNO]> plan(pomdp);
        for (int iter = 0; iter < steps; iter++) {
            // cout << "---------- Iteration " << iter+1 << " ----------" << endl;
            //cout << "Curr Belief -- ";
            //print_vector(plan.curr_belief);
            int next_action = -1;
            // t = clock();
            if (OPT) {
                next_action = plan.backup_plan(tr, err, pomdp.o, zeros, true, 50);
            }
            else {
                next_action = plan.backup_plan(tr, zeros, pomdp.o, zeros, true, 50);
            }
            assert (next_action >= 0);
            t = clock() - t;
             //cout << "Step: " << ((float) t)/CLOCKS_PER_SEC << endl;
             //cout << "next action: " << next_action << endl;

            // advance the pomdp
            //pomdp.step(0);
            pomdp.step(next_action);
            //pomdp.step(rand() % pomdp.numactions);
            //cout << "Curr Belief -- ";
            //print_vector(plan.curr_belief);
            // update beliefs
             t = clock();
            plan.belief_update_full();
             t = clock() - t;
            // cout << "Belief Update: " << ((float) t)/CLOCKS_PER_SEC << endl;
            //plan.print_points();
            //cout << "o" << endl;
            //for (int i = 0; i < TIGER_NUMSTATES; ++i)
            //{
            //for (int j = 0; j < TIGER_NUMACTIONS; ++j)
            //{
            //for (int k = 0; k < TIGER_NUMSTATES; ++k)
            //{
            //cout << plan.opt_t[i][j][k] << " ";
            //}
            //cout << "| ";
            //}
            //cout << endl;
            //}
            double res[TNS][TNA][TNS];
            // t = clock();
            initialize(pomdp, res);
            em(pomdp, res, res);
            // t = clock() - t;
             //cout << "EM: " << ((float) t)/CLOCKS_PER_SEC << endl;
            for (int x = 0; x < pomdp.numstates; x++) {
                for (int y = 0; y < pomdp.numactions; y++) {
                    for (int z = 0; z < pomdp.numstates; z++) {
                        tr[x][y][z] = res[x][y][z];
                        //tr[x][y][z] = pomdp.t[x][y][z];
                        //cout << x << "," << y << "," << z << " | " << tr[x][y][z] << endl;
                    }
                }
            }
            //cout << "err" << endl;
            //for (int x = 0; x < pomdp.numstates; x++) {
                //for (int y = 0; y < pomdp.numactions; y++) {
                    //for (int z = 0; z < pomdp.numstates; z++) {
                        //cout << x << "," << y << "," << z << " | " << err[x][y][z] << endl;
                    //}
                //}
            //}
            // t = clock();
            if (OPT) {
                //cout << pomdp.obs.size() << endl;
                double boot_tr[B][pomdp.numstates][pomdp.numactions][pomdp.numstates];
                for (int b = 0; b < B; b++) {
                    POMDP learnedpomdp;
                    learnedpomdp.set_tr(tr);
                    //learnedpomdp.set_t(t);
                    for (size_t i = 0; i < pomdp.obs.size(); ++i) {
                        learnedpomdp.step(pomdp.actions[i]);
                        //learnedpomdp.step(rand() % pomdp.numactions);
                    }
                    double new_tr[TNS][TNA][TNS];
                    initialize(pomdp, new_tr);
                    em(learnedpomdp, new_tr, new_tr);
                    for (int i = 0; i < pomdp.numstates; i++) {
                        for (int j = 0; j < pomdp.numactions; j++) {
                            for (int k = 0; k < pomdp.numstates; k++) {
                                boot_tr[b][i][j][k] = new_tr[i][j][k];
                            }
                        }
                    }
                }
                double sum[pomdp.numstates][pomdp.numactions][pomdp.numstates];
                for (int i = 0; i < pomdp.numstates; i++) {
                    for (int j = 0; j < pomdp.numactions; j++) {
                        for (int k = 0; k < pomdp.numstates; k++) {
                            sum[i][j][k] = 0;
                            err[i][j][k] = 0;
                            for (int b = 0; b < B; b++) {
                                sum[i][j][k] += boot_tr[b][i][j][k];
                            }
                            for (int b = 0; b < B; b++) {
                                err[i][j][k] += pow(boot_tr[b][i][j][k] - sum[i][j][k]/B, 2);
                            }
                            err[i][j][k] = 1.0 * sqrt(err[i][j][k]/B);
                            if (err[i][j][k] == 0)
                            {
                                err[i][j][k] = 1;
                            }
                            //err[i][j][k] = 1.0/sqrt(iter);
                            //err[i][j][k] = 1.0/iter;
                        }
                    }
                }
            }
            // t = clock() - t;
            // cout << "Bootstrap: " << ((float) t)/CLOCKS_PER_SEC << endl;

            // t = clock();
            // cout << "o" << endl;
            // for (int i = 0; i < TIGER_NUMSTATES; ++i)
            // {
            //         for (int j = 0; j < TIGER_NUMACTIONS; ++j)
            //         {
            //                 for (int k = 0; k < TIGER_NUMOBS; ++k)
            //                 {
            //                         //cout << plan.opt_z[i][j][k] << " ";
            //                         cout << o[i][j][k] << " ";
            //                 }
            //                 cout << "|";
            //         }
            //         cout << endl;
            // }
            // cout << "ci" << endl;
            // for (int i = 0; i < TIGER_NUMSTATES; ++i)
            // {
            //         for (int j = 0; j < TIGER_NUMACTIONS; ++j)
            //         {
            //                 for (int k = 0; k < TIGER_NUMOBS; ++k)
            //                 {
            //                         //cout << plan.opt_z[i][j][k] << " ";
            //                         cout << err[i][j][k] << " ";
            //                 }
            //                 cout << "|";
            //         }
            //         cout << endl;
            // }
            // cout << iter << endl;
            // t = clock() - t;
            // cout << "printing: " << ((float) t)/CLOCKS_PER_SEC << endl;
        }
        //print_vector(pomdp.obs);
        // cout << "Rewards" << endl;
        // print_vector(pomdp.rewards);
        // int last = 0;
        // for (int i = 0; i < pomdp.actions.size(); ++i)
        // {
        //         cout << pomdp.actions[i] << " " << pomdp.obs[i] << endl;
        // }
        // print_vector(pomdp.actions);
        // int act = 0;
        // for (act = 0; act < steps; act++) {
        //         if (pomdp.actions[act] == 2) {
        //                 last = act;
        //         }
        // }

        for (size_t i = 0; i < pomdp.rewards.size(); ++i)
        {
            rs[i] += pomdp.rewards[i];
            sum_rewards += pomdp.rewards[i];
        }
        // cout << "Rewards: " << sum_rewards - prev_sum << endl;
        rewardsFile << sum_rewards - prev_sum << endl;
        prev_sum = sum_rewards;
        // rept = clock() - rept;
        // cout << "One Rep: " << ((float) rept)/CLOCKS_PER_SEC << endl;
        
        // show some stats on the run
        if (1)
        {
            cout << "rewards" << endl;
            for (size_t i = 0; i < pomdp.rewards.size(); ++i)
            {
                cout << pomdp.rewards[i] << " ";
            }
            cout << endl;
            cout << "actions" << endl;
            for (size_t i = 0; i < pomdp.actions.size(); ++i)
            {
                cout << pomdp.actions[i] << " ";
            }
            cout << endl;
        }
    }
    // ofstream cout("out.txt");
    // for (size_t i = 0; i < rs.size(); ++i)
    // {
    //         rs[i] /= reps;
    //         cout << rs[i] << " ";
    // }
    // cout << endl;
    // print_vector(rs);
    cout << "Cumulative reward " << sum_rewards/reps << endl;
    rewardsFile.close();
    // outFile.close();
    return 0;
}
