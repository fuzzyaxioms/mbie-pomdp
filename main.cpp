#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include "planning.hpp"

using namespace std;

double logmul (double a, double b){
        return a + b;
}

double logadd (double a, double b){
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

#define TIGER_REWARD 10
#define TIGER_PENALTY (-100)
#define OPT true
#define FILENAME "2sensor_opt_rewards_100alpha.txt"

// ofstream outFile;
clock_t t;
clock_t rept;

struct POMDP
{
        int numstates;
        int numactions;
        int numobs;
        
        double gamma;
        double rmax;
        
        double t[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMSTATES];
        double o[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMOBS];
        double r[TIGER_NUMSTATES][TIGER_NUMACTIONS];
        
        vector<int> actions;
        vector<int> obs;
        vector<int> rewards;
        
        int curr_state;
        
        POMDP()
        : numstates(TIGER_NUMSTATES), numactions(TIGER_NUMACTIONS), numobs(TIGER_NUMOBS),
        gamma(0.99), rmax(10)
        {
                for (int i = 0; i < numstates; ++i)
                {
                        for (int j = 0; j < numactions; ++j)
                        {
                                r[i][j] = 0.0;
                                for (int k = 0; k < numstates; ++k)
                                {
                                        t[i][j][k] = 0.0;
                                }
                                for (int k = 0; k < numobs; ++k)
                                {
                                        o[i][j][k] = 0.0;
                                }
                        }
                }
                // actions are: open left, open right, listen
                // states are: left, right
                // obs are: hear left, hear right
                
                // listening is always -1
                r[0][2] = -1;
                r[1][2] = -1;

                r[0][3] = -1;
                r[1][3] = -1;

                
                // opening right door is worth TIGER_REWARD
                r[0][0] = TIGER_REWARD;
                r[1][1] = TIGER_REWARD;
                
                // opening wrong door costs TIGER_PENALTY
                r[0][1] = TIGER_PENALTY;
                r[1][0] = TIGER_PENALTY;
                
                // transitions are easy
                // listening preserves state
                t[0][2][0] = 1.0;
                t[1][2][1] = 1.0;

                t[0][3][0] = 1.0;
                t[1][3][1] = 1.0;

                // opening a door always resets
                t[0][0][0] = 0.5;
                t[0][0][1] = 0.5;
                t[1][0][0] = 0.5;
                t[1][0][1] = 0.5;
                t[0][1][0] = 0.5;
                t[0][1][1] = 0.5;
                t[1][1][0] = 0.5;
                t[1][1][1] = 0.5;
                
                // listening most of the time gets the right door
                double acc = 0.85;
                o[0][2][0] = acc;
                o[0][2][1] = 1-acc;
                o[1][2][0] = 1-acc;
                o[1][2][1] = acc;

                double acc2 = 0.9;
                o[0][3][0] = acc2;
                o[0][3][1] = 1-acc2;
                o[1][3][0] = 1-acc2;
                o[1][3][1] = acc2;

                // now after a reset, both obs have equal chance
                o[0][0][0] = 0.5;
                o[0][0][1] = 0.5;
                o[0][1][0] = 0.5;
                o[0][1][1] = 0.5;
                o[1][0][0] = 0.5;
                o[1][0][1] = 0.5;
                o[1][1][0] = 0.5;
                o[1][1][1] = 0.5;
                
                // suppose after a reset the obs actually does tell you something
                //o[0][0][0] = acc;
                //o[0][0][1] = 1-acc;
                //o[0][1][0] = acc;
                //o[0][1][1] = 1-acc;
                //o[1][0][0] = 1-acc;
                //o[1][0][1] = acc;
                //o[1][1][0] = 1-acc;
                //o[1][1][1] = acc;
                
                // start
                curr_state = rand() % numstates;
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
                curr_state = next_state;
                
                //cout << "action " << action << " " << prev_state << " -> " << next_state << endl;
        }
        void set(double (&new_o)[TNS][TNA][TNO])
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
};

void initialize(POMDP &pomdp, double (&o)[TNS][TNA][TNO])
{
        double p;
        double p2;
        double total;
        for (int i = 0; i < pomdp.numstates; ++i)
        {
                for (int j = 0; j < pomdp.numactions; ++j)
                {
                        total = 0;
                        for (int k = 0; k < pomdp.numobs; ++k)
                        {
                                p = rand();
                                //p = 0.5;
                                o[i][j][k] = p;
                                total += p;
                        }
                        for (int k = 0; k < pomdp.numobs; ++k)
                        {
                                o[i][j][k] /= total;
                        }
                }
        }
        p = (rand()*1.0 / RAND_MAX)*0.4 + 0.6;
        //p = 0.6;
        o[0][2][0] = p;
        o[0][2][1] = 1-p;
        o[1][2][0] = 1-p;
        o[1][2][1] = p;
        p2 = (rand()*1.0 / RAND_MAX)*0.4 + 0.6;
        o[0][3][0] = p2;
        o[0][3][1] = 1-p2;
        o[1][3][0] = 1-p2;
        o[1][3][1] = p2;
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

void em(POMDP &pomdp, double (&o)[TNS][TNA][TNO], double (&best_o)[TNS][TNA][TNO])
{
        double prevloglike = log(0);
        for (int restart=0; restart < 1; restart++){
                double max = 1;
                double loglike;
                double loglike2; // This is another way to calculate the log likelihood and is used as a sanity check.
                // double learned_o[TNS][TNA][TNO];
                // double o[TNS][TNA][TNO];
                // initialize(pomdp, o);
                double prev_o[TNS][TNA][TNO];
                for (int x = 0; x < pomdp.numstates; x++){
                        for (int y = 0; y < pomdp.numactions; y++){
                                for (int z = 0; z < pomdp.numobs; z++){
                                        prev_o[x][y][z] = 1;
                                }
                        }
                }
                //typedef double SVec[TNS];
                const int T = pomdp.obs.size() - 1;
                vector<SVec> alpha(T+1);
                vector<SVec> beta(T+1);
                vector<SVec> gamma(T+1);
                double gammasum [TNA];
                double obsgammasum [TNA][TNO];
                double pi [2] = {0.5, 0.5};
                for (int iters = 0; iters < 10 and max > 0.001; iters++){
                        // This should be initialized outside of the loop, but for some odd reason
                        // its value seems to change after each iteration...so we'll put it here for now.
                        max = 0;
                        for (int i = 0; i < pomdp.numstates; i++){
                                alpha[0][i] = logmul(log(pi[i]), log(o[i][pomdp.actions[0]][pomdp.obs[0]]));
                        }
                        for (int l = 1; l <= T; l++){
                                for (int j = 0; j < pomdp.numstates; j++){
                                        double sum = log(0.0);
                                        for (int i = 0; i < pomdp.numstates; i++){
                                                sum = logadd(sum, logmul(alpha[l - 1][i], log(pomdp.t[i][pomdp.actions[l]][j])));
                                        }
                                        alpha[l][j] = logmul(sum, log(o[j][pomdp.actions[l]][pomdp.obs[l]]));
                                }
                        }

                        for (int i = 0; i < pomdp.numstates; i++){
                                beta[T][i] = log(1);
                        }
                        for (int l = T - 1; l >= 0; l--){
                                for (int i = 0; i < pomdp.numstates; i++){
                                        beta[l][i] = log(0.0);
                                        for (int j = 0; j < pomdp.numstates; j++){
                                                double prod = logmul(log(pomdp.t[i][pomdp.actions[l + 1]][j]), log(o[j][pomdp.actions[l + 1]][pomdp.obs[l + 1]]));
                                                prod = logmul(prod, beta[l + 1][j]);
                                                beta[l][i] = logadd(beta[l][i], prod);
                                        }
                                }
                        }
                        loglike = log(0);
                        loglike2 = log(0);
                        for (int i = 0; i < pomdp.numstates; i++){
                                loglike = logadd(loglike, alpha[T][i]);
                                loglike2 = logadd(loglike2, logmul(logmul(beta[0][i], log(pi[i])), log(o[i][pomdp.actions[0]][pomdp.obs[0]])));
                        }

                        // loglike and loglike 2 should be (nearly) the same
                        //assert(abs((loglike - loglike2)/loglike) <= .01);

                        for (int l = 0; l <= T; l++){
                                double sum = 0;
                                for (int i = 0; i < pomdp.numstates; i++) {
                                        if (i == 0){
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

                        if (true)
                        {
                                // use rewards to set the state
                                int last_state = -1;
                                for (int l = T-1; l >= 0; --l)
                                {
                                        if (abs(pomdp.rewards[l+1] - TIGER_REWARD) < 0.001)
                                        {
                                                // opened correct door
                                                assert (pomdp.actions[l+1] == 0 or pomdp.actions[l+1] == 1);
                                                last_state = pomdp.actions[l+1];
                                        }
                                        else if (abs(pomdp.rewards[l+1] - TIGER_PENALTY) < 0.001)
                                        {
                                                // opened wrong door
                                                assert (pomdp.actions[l+1] == 0 or pomdp.actions[l+1] == 1);
                                                last_state = 1-pomdp.actions[l+1];
                                        }
                                        // set the state in stone
                                        // hardcoded for tiger
                                        if (last_state >= 0)
                                        {
                                                gamma[l][last_state] = log(1);
                                                gamma[l][1-last_state] = log(0);
                                        }
                                }
                        }
                        for (int i = 0; i < pomdp.numstates; i++){
                                // std::fill(gammasum, gammasum + 3, double(log(pomdp.numobs)));
                                // std::fill(obsgammasum[0], obsgammasum[3] + 2, double(log(1)));
                                for (int x = 0; x < pomdp.numactions; x++){
                                        // double sum = 0.0;
                                        for (int y = 0; y < pomdp.numobs; y++){
                                                // double rnd = sample_unif();
                                                // sum += rnd;
                                                // obsgammasum[x][y] = log(rnd);
                                                obsgammasum[x][y] = log(0.0);

                                        }
                                        gammasum[x] = log(0.0);
                                }
                                for (int l = 0; l <= T - 1; l++){
                                        gammasum[pomdp.actions[l]] = logadd(gammasum[pomdp.actions[l]], gamma[l][i]);
                                        obsgammasum[pomdp.actions[l]][pomdp.obs[l]] = logadd(obsgammasum[pomdp.actions[l]][pomdp.obs[l]], gamma[l][i]);
                                }
                                obsgammasum[pomdp.actions[T]][pomdp.obs[T]] = logadd(obsgammasum[pomdp.actions[T]][pomdp.obs[T]], gamma[T][i]);
                                for (int x = 0; x < pomdp.numactions; x++){
                                        double sum = 0.0;
                                        for (int y = 0; y < pomdp.numobs; y++){
                                                if (obsgammasum[x][y] == log(0.0)){
                                                        obsgammasum[x][y] = log(1.0);
                                                        sum += 1.0;      
                                                }
                                        }
                                        gammasum[x] = logadd(gammasum[x], log(sum));
                                }
                                for (int a = 0; a < pomdp.numactions; a++){
                                        if (a == pomdp.actions[T]) {
                                                gammasum[a] = logadd(gammasum[a], gamma[T][i]);
                                        }
                                        for (int z = 0; z < pomdp.numobs; z++){
                                                double delta = abs((prev_o[i][a][z] - exp(logmul(obsgammasum[a][z], -gammasum[a]))));
                                                if (delta > max){
                                                        max = delta;
                                                }
                                                o[i][a][z] = exp(logmul(obsgammasum[a][z], -gammasum[a]));
                                                prev_o[i][a][z] = exp(logmul(obsgammasum[a][z], -gammasum[a]));
                                        }
                                }
                        }
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
                        // for (int x = 0; x < pomdp.numstates; x++){
                        //         for (int y = 0; y < pomdp.numactions; y++){
                        //                 for (int z = 0; z < pomdp.numobs; z++){
                        //                         learned_o[x][y][z] = o[x][y][z];
                        //                 }
                        //         }
                        // }
                }
                if (prevloglike < loglike){
                        //cout << loglike << endl;
                        prevloglike = loglike;
                        for (int x = 0; x < pomdp.numstates; x++){
                                for (int y = 0; y < pomdp.numactions; y++){
                                        for (int z = 0; z < pomdp.numobs; z++){
                                                best_o[x][y][z] = o[x][y][z];
                                        }
                                }
                        }                
                }
                //cout << "em called" << endl;
                //for (int l = 0; l <= T; l++){
                        //cout << "sim step " << l << ": ";
                        //for (int i = 0; i < pomdp.numstates; i++) {
                                //cout << exp(gamma[l][i]) << " ";
                        //}
                        //cout << endl;
                //}
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
        srand(time(0));
        // outFile.open("outstream_100alpha.txt", ios::out | ios::app);
        cout << "hello all" << endl;

        int B = 10;
        int reps = 100;
        int steps = 20000;
        double sum_rewards = 0;
        int listen_time = 2;
        int sim_steps = 0;
        int prev_sum = 0;
        
        vector<double> rs(max(steps,sim_steps), 0.0);

        double zeros[TNS][TNA][TNS];
        double o[TNS][TNA][TNO];
        double err[TNS][TNA][TNO];
        for (int x = 0; x < TNS; x++){
                for (int y = 0; y < TNA; y++){
                        for (int z = 0; z < TNS; z++){
                                zeros[x][y][z] = 0;
                        }
                        for (int z = 0; z < TNO; z++){
                                o[x][y][z] = 0;
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
                initialize(pomdp, o);
                Planning<POMDP,double[TNS][TNA][TNS],double[TNS][TNA][TNO]> plan(pomdp);
                for (int iter = 0; iter < steps; iter++){
                        // cout << "---------- Iteration " << iter+1 << " ----------" << endl;
                        //cout << "Curr Belief -- ";
                        //print_vector(plan.curr_belief);
                        int next_action = 1;
                        // t = clock();
                        if (OPT) { 
                                next_action = plan.backup_plan(pomdp.t, zeros, o, err); 
                        }
                        else { 
                                next_action = plan.backup_plan(pomdp.t, zeros, o, zeros); 
                        }
                        t = clock() - t;
                        // cout << "Step: " << ((float) t)/CLOCKS_PER_SEC << endl;
                        // cout << "next action: " << next_action << endl;
                        
                        // advance the pomdp
                        for (int simi = 0; simi < sim_steps; ++simi)
                        {
                                //cout << "simi " << simi << ": " << pomdp.curr_state << endl;
                                if (simi % (listen_time) == 0)
                                {
                                        pomdp.step((simi/listen_time) % 2);
                                }
                                else
                                {
                                        pomdp.step(2);
                                }
                        }
                        pomdp.step(next_action);
                        //cout << "Curr Belief -- ";
                        //print_vector(plan.curr_belief);
                        // update beliefs
                        // t = clock();
                        plan.belief_update();
                        // t = clock() - t;
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
                        double res[TNS][TNA][TNO];
                        // t = clock();
                        initialize(pomdp, res);
                        em(pomdp, res, res);
                        // t = clock() - t;
                        // cout << "EM: " << ((float) t)/CLOCKS_PER_SEC << endl;
                        for (int x = 0; x < pomdp.numstates; x++){
                                for (int y = 0; y < pomdp.numactions; y++){
                                        for (int z = 0; z < pomdp.numstates; z++){
                                                o[x][y][z] = res[x][y][z];
                                                //o[x][y][z] = pomdp.o[x][y][z];
                                        }
                                }
                        }
                        // t = clock();
                        if (OPT) {
                                //cout << pomdp.obs.size() << endl;
                                double boot_o[B][pomdp.numstates][pomdp.numactions][pomdp.numobs];
                                for (int b = 0; b < B; b++){
                                        POMDP learnedpomdp;
                                        learnedpomdp.set(o);
                                        for (size_t i = 0; i < pomdp.obs.size(); ++i){
                                                learnedpomdp.step(pomdp.actions[i]);
                                                //learnedpomdp.step(rand() % pomdp.numactions);
                                        }
                                        double new_o[TNS][TNA][TNO];
                                        initialize(pomdp, new_o);
                                        em(learnedpomdp, new_o, new_o);
                                        for (int i = 0; i < pomdp.numstates; i++){
                                                for (int j = 0; j < pomdp.numactions; j++){
                                                        for (int k = 0; k < pomdp.numobs; k++){
                                                                boot_o[b][i][j][k] = new_o[i][j][k];
                                                        }
                                                }
                                        }
                                }
                                double sum[pomdp.numstates][pomdp.numactions][pomdp.numobs];
                                for (int i = 0; i < pomdp.numstates; i++){
                                        for (int j = 0; j < pomdp.numactions; j++){
                                                for (int k = 0; k < pomdp.numobs; k++){
                                                        sum[i][j][k] = 0;
                                                        err[i][j][k] = 0;
                                                        for (int b = 0; b < B; b++){
                                                                sum[i][j][k] += boot_o[b][i][j][k];
                                                        }
                                                        for (int b = 0; b < B; b++){
                                                                err[i][j][k] += pow(boot_o[b][i][j][k] - sum[i][j][k]/B, 2);
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
