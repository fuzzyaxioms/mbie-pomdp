#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <algorithm>

#define TIGER_NUMACTIONS 2
#define TIGER_NUMSTATES 2
#define TIGER_NUMOBS 2

#define NUM_BELIEFS 20

#include "planning.hpp"

using namespace std;

#define TNS TIGER_NUMSTATES
#define TNA TIGER_NUMACTIONS
#define TNO TIGER_NUMOBS

//#define TIGER_REWARD 10
//#define TIGER_PENALTY (-100)

#define USE_REWARDS false

#define SMALL_REWARD 0.5
#define BIG_REWARD 1.0
#define SUCCESS_PROB 0.05
#define OBS_SUCCESS 0.8

#define START_STATE (0)
//#define START_STATE (rand() % pomdp.numstates)

// ofstream outFile;
clock_t t;
clock_t rept;

double logmul (double a, double b) {
    return a + b;
}

double logadd (double a, double b) {
    double m = max(a, b);
    if (m == log(0.0))
    {
        return m;
    }
    else
    {
        return m + log(exp(a - m) + exp(b - m));
    }
}


void print_t(double const (&tr)[TNS][TNA][TNS])
{
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            for (int z = 0; z < TNS; ++z)
            {
                cout << x << "," << y << "," << z << " | " << tr[x][y][z] << endl;
            }
        }
    }
}

void print_r(double const (&r)[TNS][TNA])
{
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            cout << x << "," << y << " | " << r[x][y] << endl;
        }
    }
}

void copy_t(double const (&src_tr)[TNS][TNA][TNS], double (&dst_tr)[TNS][TNA][TNS])
{
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            for (int z = 0; z < TNS; ++z)
            {
                dst_tr[x][y][z] = src_tr[x][y][z];
            }
        }
    }
}

void copy_r(double const (&src_r)[TNS][TNA], double (&dst_r)[TNS][TNA])
{
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            dst_r[x][y] = src_r[x][y];
        }
    }
}

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
    vector<double> rewards;
    
    // useful for debugging
    vector<int> states;

    int curr_state;

    POMDP()
        : numstates(TNS), numactions(TNA), numobs(TNO),
          gamma(1.00), rmax(BIG_REWARD)
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
                //double p = sample_unif();
                double p = sample_gamma();
                //double p = 1.0;
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
// also output log likelihood
// also give confidence intervals that come from the expected counts
double em(POMDP &pomdp, double (&tr)[TNS][TNA][TNS], double (&best_tr)[TNS][TNA][TNS], double (&err)[TNS][TNA][TNS], double (&est_r)[TNS][TNA], double (&opt_r)[TNS][TNA])
{
    double max = 1;
    const int T = pomdp.obs.size() - 2;
    if (T < 0)
    {
        return log(1.0);
    }
    vector<SVec> alpha(T+1);
    vector<SVec> beta(T+1);
    vector<SVec> gamma(T+1);
    
    // at time t, the current state is pomdp.states[t]
    // pomdp.rewards[t] is the reward for the previous state
    // pomdp.obs[t] is the current obs
    // pomdp.actions[t] is the previous action
    
    double pi [2] = {0.5, 0.5};
    const int num_iters = 10;
    for (int iters = 0; iters < num_iters and max > 0.001; iters++) {
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
        
        // inferring reward values
        double ex_reward[TNS][TNA];
        
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
                ex_reward[x][y] = 0.0;
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
            
            // calculate gamma_action_sum
            for (int x = 0; x < pomdp.numstates; ++x)
            {
                
                // update only the entries corresponding to the action at this timestep
                gamma_action_sum[x][cact] = logadd(gamma[l][x],gamma_action_sum[x][cact]);
                
                for (int z = 0; z < pomdp.numstates; ++z)
                {
                    // numerator of the expression
                    // instead of calling logmul I just use +
                    double top = alpha[l][x] + log(tr[x][cact][z]) + log(pomdp.o[z][cact][pomdp.obs[l+1]]) + beta[l+1][z];
                    sum = logadd(sum, top);
                    temp_vals[x][z] = top;
                }
            }
            
            // calculate epsilon_sum
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
            
            // calculate ex_reward
            for (int x = 0; x < pomdp.numstates; ++x)
            {
                // weighted by the belief prob
                ex_reward[x][cact] += pomdp.rewards[l+1] * exp(gamma[l][x]);
            }
        }
        
        // now it's easy to compute the estimated probs using the sum variables above
        // can also get the fake confidence intervals from the expected counts
        for (int x = 0; x < pomdp.numstates; ++x)
        {
            for (int y = 0; y < pomdp.numactions; ++y)
            {
                // compute the confidence intervals
                double const confidence_alpha = 0.05;
                double fake_count = exp(gamma_action_sum[x][y]);
                double ci_radius = fake_count > 0.0 ? sqrt((0.5/fake_count)*log (2.0/confidence_alpha)) : 1.0;
                
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
                    err[x][y][z] = ci_radius;
                    //err[x][y][z] = 1;
                }
                // normalize to fix stuff up
                // not sure why this isn't already normalized - they should be
                for (int z = 0; z < pomdp.numstates; ++z)
                {
                    tr[x][y][z] /= sum;
                }
                
                // compute expected reward
                if (gamma_action_sum[x][y] == log(0.0))
                {
                    // if no data, then rmax
                    opt_r[x][y] = pomdp.rmax;
                    est_r[x][y] = pomdp.rmax;
                }
                else
                {
                    opt_r[x][y] = min(ex_reward[x][y] / exp(gamma_action_sum[x][y]) + ci_radius, pomdp.rmax);
                    est_r[x][y] = min(ex_reward[x][y] / exp(gamma_action_sum[x][y]), pomdp.rmax);
                }
            }
        }
        
        // for debugging
        if (0 and iters == num_iters-1)
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
            print_t(tr);
        }
    }
    for (int x = 0; x < pomdp.numstates; x++) {
        for (int y = 0; y < pomdp.numactions; y++) {
            for (int z = 0; z < pomdp.numstates; z++) {
                best_tr[x][y][z] = tr[x][y][z];
            }
        }
    }
    
    // calculate log likelihood from alpha
    double loglikelihood = log(0.0);
    for (int x = 0; x < pomdp.numstates; ++x)
    {
        loglikelihood = logadd(alpha[T][x],loglikelihood);
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
    return loglikelihood;
}


double best_em(POMDP &pomdp, double (&tr)[TNS][TNA][TNS], double (&best_tr)[TNS][TNA][TNS], double (&err)[TNS][TNA][TNS], double (&est_r)[TNS][TNA], double (&opt_r)[TNS][TNA])
{
    double max_ll = log(0.0);
    double curr_best_tr[TNS][TNA][TNS];
    double curr_best_err[TNS][TNA][TNS];
    double curr_best_est_r[TNS][TNA];
    double curr_best_opt_r[TNS][TNA];
    
    for (int x = 0; x < TNS; x++) {
        for (int y = 0; y < TNA; y++) {
            curr_best_est_r[x][y] = 0.0;
            curr_best_opt_r[x][y] = 0.0;
            for (int z = 0; z < TNS; z++) {
                curr_best_tr[x][y][z] = 0;
                curr_best_err[x][y][z] = 0;
            }
        }
    }
    
    for (int i = 0; i < 100; ++i)
    {
        initialize(pomdp, tr);
        double curr_ll = em(pomdp, tr, best_tr, err, est_r, opt_r);
        if (curr_ll > max_ll)
        {
            max_ll = curr_ll;
            copy_t(best_tr, curr_best_tr);
            copy_t(err, curr_best_err);
            copy_r(est_r, curr_best_est_r);
            copy_r(opt_r, curr_best_opt_r);
        }
    }
    copy_t(curr_best_tr, best_tr);
    copy_t(curr_best_err, err);
    copy_r(curr_best_est_r, est_r);
    copy_r(curr_best_opt_r, opt_r);
    return max_ll;
}


// find a good number for the number of belief points and times to iterate
// looks like 30 iterations is good enough to find the optimal policy with a big gap
// but to be safe let's go with 40 since that works with worse sensors as well
// rewards are 1 and 3, and obs success is 0.8 and success prob is 0.05

// but for some reason once I go to 100 belief points, it can't find
// the optimal policy with estimated transitions and big confidence intervals
// the reason is that the agent is having a hard time finding out that it's good
// to stay in the right state, em keeps estimating that going right in the right state
// gets you back to the left state, and the ci is too small to allow the optimism
// to swing it the other way. With the way things are set up, if you estimate
// that it's 0.5 to stay in the right state when taking the right action, the agent
// is better off not trying to get to the right state. To fix this, I increase
// the size of the confidence intervals, and increase big reward slightly so that
// there is wiggle room for the agent to be optimistic enough to want to try to
// go to the right state.
void find_planning_params()
{
    //srand(0);
    srand(time(0));
    POMDP pomdp;
    Planning<POMDP,double[TNS][TNA][TNS],double[TNS][TNA][TNO],double[TNS][TNA]> plan(pomdp);
    
    double o_zeros[TNS][TNA][TNO];
    double err[TNS][TNA][TNS];
    double tr[TNS][TNA][TNS];
    for (int x = 0; x < TNS; x++) {
        for (int y = 0; y < TNA; y++) {
            for (int z = 0; z < TNO; z++) {
                o_zeros[x][y][z] = 0;
            }
            for (int z = 0; z < TNS; z++) {
                tr[x][y][z] = 0;
                err[x][y][z] = 0;
            }
        }
    }
    
    // let's make a purposefully bad transition matrix but with
    tr[0][0][0] = 1.0; // stay in the same state
    tr[0][0][1] = 0.0;
    
    err[0][0][0] = 0.01; // a small chance to include more enticing probs
    err[0][0][1] = 0.01;
    
    tr[0][1][0] = 0.95; // hard to go right
    tr[0][1][1] = 0.05;
    
    //err[0][1][0] = 0.05; // a small chance to include correct probs
    //err[0][1][1] = 0.05;
    
    tr[1][0][0] = 1.0; // easily go back to left
    tr[1][0][1] = 0.0;
    
    tr[1][1][0] = 0.00; // stay right
    tr[1][1][1] = 1.00;
    
    //err[1][1][0] = 0.5; // wrong estimate possible
    //err[1][1][1] = 0.5;
    
    int next_action = plan.backup_plan(tr, err, pomdp.o, o_zeros, pomdp.r, true, 40);
    
    print_t(plan.opt_t);
    
    assert (next_action >= 0);
    plan.print_points();
}

// testing the correctness of em
// seems to work now

// need to be careful when estimating something to have probability 0 or 1
// because that might get rid of all variance when running bootstrap
void test_em()
{
    srand(0);
    //srand(time(0));

    //int B = 0;
    int steps = 300;

    double tr[TNS][TNA][TNS];
    double err[TNS][TNA][TNS];
    double est_r[TNS][TNA];
    double opt_r[TNS][TNA];
    for (int x = 0; x < TNS; x++) {
        for (int y = 0; y < TNA; y++) {
            est_r[x][y] = 0.0;
            opt_r[x][y] = 0.0;
            for (int z = 0; z < TNS; z++) {
                tr[x][y][z] = 0;
                err[x][y][z] = 0;
            }
        }
    }
    
    POMDP pomdp;
    
    // cout << "---------- Iteration " << iter+1 << " ----------" << endl;
    // advance the pomdp
    for (int i = 0; i < steps; ++i)
    {
        int next_action = sample_unif() > 0.5;
        //int next_action = (i / 10) % 2;
        assert (next_action >= 0);
        pomdp.step(next_action);
    }
    // show some stats on the run
    cout << "states" << endl;
    for (size_t i = 0; i < pomdp.rewards.size(); ++i)
    {
        cout << pomdp.states[i] << " ";
    }
    cout << endl;
    cout << "actions" << endl;
    for (size_t i = 0; i < pomdp.actions.size(); ++i)
    {
        cout << pomdp.actions[i] << " ";
    }
    cout << endl;
    cout << "rewards" << endl;
    for (size_t i = 0; i < pomdp.rewards.size(); ++i)
    {
        cout << pomdp.rewards[i] << " ";
    }
    cout << endl;
    
    srand(time(0));
    initialize(pomdp, tr);
    //double loglikelihood = em(pomdp, tr, tr, err, est_r, opt_r);
    double loglikelihood = best_em(pomdp, tr, tr, err, est_r, opt_r);
    cout << "ll = " << loglikelihood << endl;
    
    if (0)
    {
        // let's make a purposefully bad transition matrix
        tr[0][0][0] = 1.0; // stay in the same state
        tr[0][0][1] = 0.0;
        
        tr[0][1][0] = 1.0; // hard to go right
        tr[0][1][1] = 0.0;
        
        tr[1][0][0] = 1.0; // easily go back to left
        tr[1][0][1] = 0.0;
        
        tr[1][1][0] = 1.0; // stay right
        tr[1][1][1] = 0.0;
    }
    
    cout << "esimated transitions" << endl;
    print_t(tr);
    cout << "esimated transitions ci" << endl;
    print_t(err);
    cout << "estimated rewards" << endl;
    print_r(est_r);
    cout << "estimated upper bound rewards" << endl;
    print_r(opt_r);
}

void test_opt(bool use_opt, char const *reward_out, double decay = -1)
{
    int seed = 0;
    //int seed = 1393009854;
    //int seed = 1393011461;
    //int seed = time(0);
    srand(seed);
    //srand();
    // outFile.open("outstream_100alpha.txt", ios::out | ios::app);
    cout << "hello all" << endl;

    //int B = 0;
    int reps = 10;
    int steps = 1000;
    double sum_rewards = 0;

    vector<double> rs(steps, 0.0);

    double zeros[TNS][TNA][TNO];
    double tr[TNS][TNA][TNS];
    double err[TNS][TNA][TNS];
    double est_r[TNS][TNA];
    double opt_r[TNS][TNA];
    for (int x = 0; x < TNS; x++) {
        for (int y = 0; y < TNA; y++) {
            est_r[x][y] = BIG_REWARD;
            opt_r[x][y] = BIG_REWARD;
            for (int z = 0; z < TNO; z++) {
                zeros[x][y][z] = 0;
            }
            for (int z = 0; z < TNS; z++) {
                tr[x][y][z] = 0;
                err[x][y][z] = 1;
            }
        }
    }

    for (int rep = 0; rep < reps; ++rep)
    {
        rept = clock();
        // cout << rep << endl;
        POMDP pomdp;
        initialize(pomdp, tr);
        Planning<POMDP,double[TNS][TNA][TNS],double[TNS][TNA][TNO],double[TNS][TNA]> plan(pomdp);
        for (int iter = 0; iter < steps; iter++) {
             //cout << "---------- Iteration " << iter+1 << " ----------" << endl;
            //cout << "Curr Belief -- ";
            //print_vector(plan.curr_belief);
            int next_action = -1;
            // t = clock();
            if (use_opt) {
                next_action = plan.backup_plan(tr, err, pomdp.o, zeros, opt_r, true, 40);
            }
            else {
                next_action = plan.backup_plan(tr, zeros, pomdp.o, zeros, est_r, true, 40);
            }
            // egreedy random action with decay
            if (decay > 0.0)
            {
                double ep_chance = 1.0 / (iter/decay + 1.0);
                if (sample_unif() < ep_chance)
                {
                    // egreedy to do actions
                    next_action = rand() % pomdp.numactions;
                }
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
            
            // debug information
            if (0)
            {
                cout << "-------------------- Iteration " << iter << " --------------------" << endl;
                cout << "(s,a,r) = " << pomdp.states.back() << "," << pomdp.actions.back() << "," << pomdp.rewards.back() << endl;
                cout << "Curr Belief -- ";
                print_vector(plan.curr_belief);
                cout << "estimated transitions" << endl;
                print_t(tr);
                cout << "optimistic transitions" << endl;
                print_t(plan.opt_t);
                cout << "estimated cis" << endl;
                print_t(err);
                cout << "estimated rewards" << endl;
                print_r(est_r);
                cout << "optimistic rewards" << endl;
                print_r(opt_r);
                cout << "-------------------- Iteration " << iter+1 << " --------------------" << endl;
            }
            
            // update beliefs
             //t = clock();
            plan.belief_update_full();
             //t = clock() - t;
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
            //double res[TNS][TNA][TNS];
            // t = clock();
            initialize(pomdp, tr);
            //em(pomdp, tr, tr, err, est_r, opt_r);
            best_em(pomdp, tr, tr, err, est_r, opt_r);
            // t = clock() - t;
             //cout << "EM: " << ((float) t)/CLOCKS_PER_SEC << endl;
        }

        for (size_t i = 0; i < pomdp.rewards.size(); ++i)
        {
            rs[i] += pomdp.rewards[i];
            sum_rewards += pomdp.rewards[i];
        }
        // cout << "Rewards: " << sum_rewards - prev_sum << endl;
        rept = clock() - rept;
        cout << "One Rep: " << ((float) rept)/CLOCKS_PER_SEC << endl;
        
        // show some stats on the run
        if (1)
        {
            cout << "states" << endl;
            for (size_t i = 0; i < pomdp.rewards.size(); ++i)
            {
                cout << pomdp.states[i] << " ";
            }
            cout << endl;
            cout << "actions" << endl;
            for (size_t i = 0; i < pomdp.actions.size(); ++i)
            {
                cout << pomdp.actions[i] << " ";
            }
            cout << endl;
            cout << "rewards" << endl;
            for (size_t i = 0; i < pomdp.rewards.size(); ++i)
            {
                cout << pomdp.rewards[i] << " ";
            }
            cout << endl;
        }
    }
    ofstream output_r(reward_out);
    for (size_t i = 0; i < rs.size(); ++i)
    {
        rs[i] /= reps;
        output_r << rs[i] << endl;
    }
    output_r << sum_rewards/reps << endl;
    output_r.close();
    
    if (1)
    {
        cout << "avg rewards" << endl;
        for (size_t i = 0; i < rs.size(); ++i)
        {
            cout << rs[i] << " ";
        }
        cout << endl;
    }
    
    cout << "Cumulative reward " << sum_rewards/reps << endl;
    cout << "seed = " << seed << endl;
}

int main ()
{
    //find_planning_params();
    
    test_em();
    
    //test_opt(true, "optimistic_rewards.txt");
    
    //int fs[] = {100};
    //for (int f: fs)
    //{
        //test_opt(false, "mean_rewards.txt", -1);
    //}
    
    // let's optimizie cumulative reward over the egreedy param for 20 points and 10 reps and 400 steps
    
    
    return 0;
}
