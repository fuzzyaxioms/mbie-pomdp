#define ARMA_DONT_USE_WRAPPER

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <armadillo>

#define NUM_BELIEFS 20
#define EM_USE_RO false
#define EST_T false
#define EST_O true
#define EST_RO (EM_USE_RO and false)
#define EST_R false

#define POMDP POMDP_MSTiger
#define TIGER_NUMACTIONS 4
#define TIGER_NUMSTATES 2
#define TIGER_NUMOBS 2
#define TNR 3
#define START_STATE (0)
#define BIG_REWARD 10
#define SMALL_REWARD 0.5 // not used
#define SUCCESS_PROB 0.1 // not used
#define OBS_SUCCESS 0.85
#define ACC1 0.75
#define ACC2 0.82
#define ACC3 0.9
#define LEARNING_EASE 0.3
#define REWARD_GAP 110.0

#include "planningorig.hpp"
#include "mom.cpp"

using namespace std;
using namespace arma;

#define TNS TIGER_NUMSTATES
#define TNA TIGER_NUMACTIONS
#define TNO TIGER_NUMOBS

//#define TIGER_REWARD 10
//#define TIGER_PENALTY (-100)

// before it was 1.0 and 2.0 the rewards, success prob was 0.05 and obs success was 0.8
// now it's 0.5 and 1.0, then 0.1, then 0.9

// ofstream outFile;
clock_t t;
clock_t rept;

double ACC[8] = {0.82, 0.83, 0.84, 0.9, 0.84, 0.86, 0.9, 0.91};

#define TOOSMALL (1e-50)

const double log_zero = log(0.0);
// const double scale = 0.99;

inline double mylogmul (double a, double b) {
    return a + b;
}

inline double mylogadd (double a, double b) {
    double m = max(a, b);
    if (m == log_zero)
    {
        return m;
    }
    else
    {
        return m + log(exp(a - m) + exp(b - m));
    }
}

template <class T, size_t A, size_t B, size_t C>
void print_matrix(T const (&arr)[A][B][C])
{
    for (size_t x = 0; x < A; ++x)
    {
        for (size_t y = 0; y < B; ++y)
        {
            for (size_t z = 0; z < C; ++z)
            {
                cout << x << "," << y << "," << z << " | " << fixed << setw(9) << arr[x][y][z] << endl;
            }
        }
    }
}

template <class T, size_t A, size_t B>
void print_matrix(T const (&arr)[A][B])
{
    for (size_t x = 0; x < A; ++x)
    {
        for (size_t y = 0; y < B; ++y)
        {
            cout << x << "," << y << " | " << fixed << setw(9) << arr[x][y] << endl;
        }
    }
}

template <class T, size_t A, size_t B, size_t C>
void copy_matrix(T const (&src)[A][B][C], T (&dst)[A][B][C])
{
    for (size_t x = 0; x < A; ++x)
    {
        for (size_t y = 0; y < B; ++y)
        {
            for (size_t z = 0; z < C; ++z)
            {
                dst[x][y][z] = src[x][y][z];
            }
        }
    }
}

template <class T, size_t A, size_t B>
void copy_matrix(T const (&src)[A][B], T (&dst)[A][B])
{
    for (size_t x = 0; x < A; ++x)
    {
        for (size_t y = 0; y < B; ++y)
        {
            dst[x][y] = src[x][y];
        }
    }
}

template <class T, size_t A, size_t B, size_t C>
void zero_out(T (&arr)[A][B][C])
{
    for (size_t i = 0; i < A; ++i)
    {
        for (size_t j = 0; j < B; ++j)
        {
            for (size_t k = 0; k < C; ++k)
            {
                arr[i][j][k] = 0;
            }
        }
    }
}

template <class T, size_t A, size_t B>
void zero_out(T (&arr)[A][B])
{
    for (size_t i = 0; i < A; ++i)
    {
        for (size_t j = 0; j < B; ++j)
        {
            arr[i][j] = 0;
        }
    }
}

template <class T, size_t A, size_t B, size_t C>
void print_both(T const (&arr)[A][B][C], T const (&err)[A][B][C])
{
    for (size_t x = 0; x < A; ++x)
    {
        for (size_t y = 0; y < B; ++y)
        {
            for (size_t z = 0; z < C; ++z)
            {
                cout << x << "," << y << "," << z << " |" << fixed << setw(12)
                    << arr[x][y][z] << setw(12) << err[x][y][z] << endl;
            }
        }
    }
}

template <class T, size_t A, size_t B>
void print_both(T const (&arr)[A][B], T const (&err)[A][B])
{
    for (size_t x = 0; x < A; ++x)
    {
        for (size_t y = 0; y < B; ++y)
        {
            cout << x << "," << y<< " |" << fixed << setw(12)
                << arr[x][y] << setw(12) << err[x][y] << endl;
        }
    }
}

struct POMDP_MSTiger
{
    int numstates;
    int numactions;
    int numobs;

    double gamma;
    double rmax;

    double t[TNS][TNA][TNS];
    double o[TNS][TNA][TNO];
    double r[TNS][TNA];
    
    // rewards as observations
    // there are three of them 0, SMALL_REWARD, and BIG_REWARD
    double ro[TNS][TNA][TNR];
    // mapping from reward obs to reward
    double ro_map[TNR];

    vector<int> actions;
    vector<int> obs;
    vector<double> rewards;
    
    // keep track of reward obs
    vector<int> reward_obs;
    
    // useful for debugging
    vector<int> states;

    int curr_state;

    POMDP_MSTiger()
        : numstates(TNS), numactions(TNA), numobs(TNO),
          gamma(0.99), rmax(BIG_REWARD)
    {
        // actions are: go left or go right
        // states are: left, right
        // obs are: hear left, hear right
        // obs are fixed

        // the only rewards are going left in left and going right in right
        r[0][0] = 10;
        r[0][1] = -100;

        r[1][0] = -100;
        r[1][1] = 10;
        
        for (int i = 2; i < TNA; i++)
        {
            r[0][i] = -1;
        }
        
        ro_map[0] = 10;
        ro_map[1] = -100;
        ro_map[2] = -1;
        
        // reflected in the reward obs
        ro[0][0][0] = 1.0; // correctly opening the door
        ro[0][0][1] = 0.0;
        ro[0][0][2] = 0.0;
        
        ro[0][1][0] = 0.0;
        ro[0][1][1] = 1.0; // wrongly opening the door
        ro[0][1][2] = 0.0;
        
        ro[1][0][0] = 0.0;
        ro[1][0][1] = 1.0; // wrongly opening the door
        ro[1][0][2] = 0.0;
        
        ro[1][1][0] = 1.0; // correctly opening the door
        ro[1][1][1] = 0.0;
        ro[1][1][2] = 0.0;

        for (int i = 2; i < TNA; i++)
        {
            ro[0][i][0] = 0.0;
            ro[0][i][1] = 0.0;
            ro[0][i][2] = 1.0; // listening gives -1

            ro[1][i][0] = 0.0;
            ro[1][i][1] = 0.0;
            ro[1][i][2] = 1.0; // listening gives -1
        }

        t[0][0][0] = 0.6; // opening door resets
        t[0][0][1] = 0.4;
        
        t[0][1][0] = 0.4; // opening door resets
        t[0][1][1] = 0.6;
        
        t[1][0][0] = 0.4; // opening door resets
        t[1][0][1] = 0.6;
        
        t[1][1][0] = 0.6; // opening door resets
        t[1][1][1] = 0.4;

        for (int i = 2; i < TNA; i++)
        {
            t[0][i][0] = 1.0; // listening stays
            t[0][i][1] = 0.0;

            t[1][i][0] = 0.0; // listening stays
            t[1][i][1] = 1.0;
        }

        // listening gives mostly correct
        o[0][0][0] = 0.7;
        o[0][0][1] = 0.3;
        
        o[0][1][0] = 0.3;
        o[0][1][1] = 0.7;
        
        o[1][0][0] = 0.3;
        o[1][0][1] = 0.7;
        
        o[1][1][0] = 0.7;
        o[1][1][1] = 0.3;
        
        for (int i = 2; i < TNA; i++)
        {
            o[0][i][0] = ACC[i-2];
            o[0][i][1] = 1.0 - ACC[i-2];

            o[1][i][0] = 1.0 - ACC[i-2];
            o[1][i][1] = ACC[i-2];
        }

        // start
        curr_state = sample_int(0, TNS-1);
    }
    
    void step(int action)
    {
        assert(action >= 0 and action < numactions);
        
        int prev_state = curr_state;
        (void)prev_state;

        // advance to the next state
        double accum = 0.0;
        double target = sample_unif();
        int next_state = -1;
        for (int i = 0; i < TNS; ++i)
        {
            accum += t[curr_state][action][i];
            if (accum >= target)
            {
                next_state = i;
                break;
            }
        }
        assert(next_state >=0 and next_state < TNS);

        // sample an obs
        int new_obs = -1;
        accum = 0.0;
        target = sample_unif();
        for (int i = 0; i < TNO; ++i)
        {
            accum += o[next_state][action][i];
            if (accum >= target)
            {
                new_obs = i;
                break;
            }
        }
        assert(new_obs >= 0 and new_obs < TNO);
        
        // sample a reward
        int new_reward_obs = -1;
        accum = 0.0;
        target = sample_unif();
        for (int i = 0; i < TNR; ++i)
        {
            accum += ro[curr_state][action][i];
            if (accum >= target)
            {
                new_reward_obs = i;
                break;
            }
        }
        assert(new_reward_obs >= 0 and new_reward_obs < TNR);
        
        // update the stuff
        actions.push_back(action);
        obs.push_back(new_obs);
        reward_obs.push_back(new_reward_obs);
        rewards.push_back(ro_map[new_reward_obs]);
        states.push_back(curr_state);
        curr_state = next_state;

        //cout << "action " << action << " " << prev_state << " -> " << next_state << endl;
    }

    void reset_nonlisten_params(double (&new_t)[TNS][TNA][TNS], double (&err_t)[TNS][TNA][TNS], double (&new_o)[TNS][TNA][TNO], double (&err_o)[TNS][TNA][TNO])
    {
        for (int x = 0; x < TNS; ++x)
        {
            for (int y = 0; y < 2; ++y)
            {
                for (int z = 0; z < TNS; ++z)
                {
                    new_t[x][y][z] = t[x][y][z];
                    err_t[x][y][z] = 0;
                }
                for (int z = 0; z < TNO; ++z)
                {
                    new_o[x][y][z] = o[x][y][z];
                    err_o[x][y][z] = 0;
                }
            }
        }
    }

    void new_episode()
    {
        curr_state = sample_int(0, TNS - 1);
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

// initialize all parameters
void initialize(POMDP &pomdp, double (&est_t)[TNS][TNA][TNS], double (&est_o)[TNS][TNA][TNO], double (&est_ro)[TNS][TNA][TNR])
{
    // completely random initialization
    for (int i = 0; i < TNS; ++i)
    {
        for (int j = 0; j < TNA; ++j)
        {
            double total = 0;
            for (int k = 0; k < TNS; ++k)
            {
                double p = sample_unif() + 0.0000000001;
                //double p = sample_gamma() + 0.0000000001;
                //double p = 1.0;
                est_t[i][j][k] = p;
                total += p;
            }
            for (int k = 0; k < TNS; ++k)
            {
                est_t[i][j][k] /= total;
            }
            total = 0;
            for (int k = 0; k < TNO; ++k)
            {
                if (j < 2)
                {
                    est_o[i][j][k] = pomdp.o[i][j][k];
                }
                else
                {
                    double p = sample_unif() + 0.0000000001;
                    //double p = sample_gamma() + 0.0000000001;
                    //double p = 1.0;
                    est_o[i][j][k] = p;
                    total += p;
                }
            }
            if (j >= 2)
            {
                for (int k = 0; k < TNO; ++k)
                {
                    est_o[i][j][k] /= total;
                }
            }
            
            total = 0;
            for (int k = 0; k < TNR; ++k)
            {
                double p = sample_unif() + 0.0000000001;
                //double p = sample_gamma() + 0.0000000001;
                //double p = 1.0;
                est_ro[i][j][k] = p;
                total += p;
            }
            for (int k = 0; k < TNR; ++k)
            {
                est_ro[i][j][k] /= total;
            }
        }
    }
    
    // maybe use prior information in initialization
    if (0 and is_same<POMDP,POMDP_MSTiger>::value)
    {
        // skew prior for observations towards the correct one
        // listening gives mostly correct
        est_o[0][0][0] = 0.6;
        est_o[0][0][1] = 0.4;
        
        est_o[1][0][0] = 0.4;
        est_o[1][0][1] = 0.6;
        
        // litening makes you probably stay where you are
        //est_t[0][0][0] = 0.6;
        //est_t[0][0][1] = 0.4;
        
        //est_t[1][0][0] = 0.4;
        //est_t[1][0][1] = 0.6;
        // it seems like with the prior for obs, the transitions are always skewed correctly
        
        // opening the correct door gives better reward
        est_ro[0][1][0] = 0.2;
        est_ro[0][1][1] = 0.2;
        est_ro[0][1][2] = 0.6;
        
        est_ro[1][2][0] = 0.2;
        est_ro[1][2][1] = 0.2;
        est_ro[1][2][2] = 0.6;
        
        // opening the wrong door gives bad reward
        est_ro[0][2][0] = 0.2;
        est_ro[0][2][1] = 0.6;
        est_ro[0][2][2] = 0.2;
        
        est_ro[1][1][0] = 0.2;
        est_ro[1][1][1] = 0.6;
        est_ro[1][1][2] = 0.2;
    }
    
    // parameters not estimated will be set to true values
    if (not EST_T)
    {
        copy_matrix(pomdp.t, est_t);
    }
    if (not EST_O)
    {
        copy_matrix(pomdp.o, est_o);
    }
    if (not EST_RO)
    {
        copy_matrix(pomdp.ro, est_ro);
    }
}

// uses reward as obs
double em(POMDP &pomdp, double (&est_t)[TNS][TNA][TNS], double (&err_t)[TNS][TNA][TNS], double (&est_o)[TNS][TNA][TNO], double (&err_o)[TNS][TNA][TNO], double (&est_ro)[TNS][TNA][TNR], double (&err_ro)[TNS][TNA][TNR], double (&est_r)[TNS][TNA], double (&opt_r)[TNS][TNA], int numtriples[TNA - 2], double scale_t = 1.0, double scale_o = 1.0, double scale_ro = 1.0, double scale_r = 1.0)
{
    // at time t, the current state is pomdp.states[t]
    // pomdp.rewards[t] is the reward for the current state
    // pomdp.reward_obs[t] is the reward obs for the current state
    // pomdp.obs[t-1] is the current obs
    // pomdp.actions[t] is the current action
    
    // the number of time steps that have occurred
    // the indices go from 0 to T-1
    const int T = pomdp.states.size();
    
    if (T <= 0)
    {
        return 0;
    }
    
    double alpha[T][TNS];
    double beta[T][TNS];
    double gamma[T][TNS];
    
    double pi [2] = {0.5, 0.5};
    
    const int num_iters = 10;
    for (int iters = 0; iters < num_iters; ++iters)
    {
        // initialize the base cases of alpha and beta
        for (int i = 0; i < TNS; ++i)
        {
            if (EM_USE_RO)
            {
                alpha[0][i] = est_ro[i][pomdp.actions[0]][pomdp.reward_obs[0]] * pi[i];
            }
            else
            {
                alpha[0][i] = pi[i];
            }
            beta[T-1][i] = 1.0;
        }
        
        // recursively build up alpha and beta from previous values
        for (int t = 1; t < T; ++t)
        {
            // alpha goes forward and beta goes backward
            int ai = t;
            int bi = T-t-1;
            // to normalize alpha and beta
            double a_denom = 0;
            double b_denom = 0;
            // do the recursive update
            for (int i = 0; i < TNS; ++i) {
                double asum = 0;
                beta[bi][i] = 0;
                for (int j = 0; j < TNS; ++j)
                {
                    asum += alpha[ai-1][j]*est_t[j][pomdp.actions[ai-1]][i];
                    if (EM_USE_RO)
                    {
                        beta[bi][i] += beta[bi+1][j]*est_t[i][pomdp.actions[bi]][j]*est_o[j][pomdp.actions[bi]][pomdp.obs[bi]]*est_ro[j][pomdp.actions[bi+1]][pomdp.reward_obs[bi+1]];
                    }
                    else
                    {
                        beta[bi][i] += beta[bi+1][j]*est_t[i][pomdp.actions[bi]][j]*est_o[j][pomdp.actions[bi]][pomdp.obs[bi]];
                    }
                }
                if (EM_USE_RO)
                {
                    alpha[ai][i] = asum * est_o[i][pomdp.actions[ai-1]][pomdp.obs[ai-1]] * est_ro[i][pomdp.actions[ai]][pomdp.reward_obs[ai]];
                }
                else
                {
                  alpha[ai][i] = asum * est_o[i][pomdp.actions[ai-1]][pomdp.obs[ai-1]];
                }
                
                a_denom += alpha[ai][i];
                b_denom += beta[bi][i];
            }
            
            assert(a_denom > TOOSMALL);
            assert(b_denom > TOOSMALL);
            
            // normalize alpha and beta
            for (int i = 0; i < TNS; ++i)
            {
                alpha[ai][i] /= a_denom;
                beta[bi][i] /= b_denom;
            }
        }
        
        // calculate gamma
        for (int t = 0; t < T; ++t) {
            double sum = 0;
            for (int i = 0; i < TNS; ++i) {
                gamma[t][i] = alpha[t][i] * beta[t][i];
                sum += gamma[t][i];
            }
            assert(sum > 0);
            for (int i = 0; i < TNS; i++) {
                gamma[t][i] /= sum;
            }
        }
        
        // now to infer transitions
        // gamma_action_sum is the expected number of times (s,a) occurred
        // the _less is for using all except the last experience
        // the _prev is for current action and next state counts
        double gamma_action_sum_less[TNS][TNA];
        double gamma_action_sum_prev[TNS][TNA];
        double gamma_action_sum[TNS][TNA];
        
        // the expected number of times (s,a,s') occurred
        double xi_sum[TNS][TNA][TNS];
        
        // the expected number of times obs occurred for (s',a)
        double obs_sum[TNS][TNA][TNO];
        
        // the expected number of times reward ob occurred for (s,a)
        double rho_sum[TNS][TNA][TNR];
        
        // inferring reward values
        double ex_reward[TNS][TNA];
        
        // initialize the expected counts to all zeros
        for (int x = 0; x < TNS; ++x)
        {
            for (int y = 0; y < TNA; ++y)
            {
                gamma_action_sum_prev[x][y] = 0;
                gamma_action_sum_less[x][y] = 0;
                gamma_action_sum[x][y] = 0;
                for (int z = 0; z < TNS; ++z)
                {
                    xi_sum[x][y][z] = 0;
                }
                for (int z = 0; z < TNO; ++z)
                {
                    obs_sum[x][y][z] = 0;
                }
                for (int r = 0; r < TNR; ++r)
                {
                    rho_sum[x][y][r] = 0;
                }
                ex_reward[x][y] = 0.0;
            }
        }
        
        // calculate gamma_action_sum variables
        for (int t = 0; t < T; ++t)
        {
            // current action
            int cact = pomdp.actions[t];
            
            // calculate gamma_action_sum
            for (int x = 0; x < TNS; ++x)
            {
                // update only the entries corresponding to the action at this timestep
                gamma_action_sum[x][cact] += gamma[t][x];
                if (t < T-1)
                {
                    gamma_action_sum_less[x][cact] += gamma[t][x];
                    gamma_action_sum_prev[x][cact] += gamma[t+1][x];
                }
            }
        }
        
        // some parameters for the confidence intervals
        double const confidence_alpha = 0.01;
        
        // estimate the transitions
        if (EST_T)
        {
            for (int t = 0; t < T-1; ++t)
            {
                // sum for normalization of the xi
                double sum = 0;
                // the numerator of the xi
                double temp_vals[TNS][TNS];
                // current action
                int cact = pomdp.actions[t];
                
                // calculate gamma_action_sum
                for (int x = 0; x < TNS; ++x)
                {
                    // update only the entries corresponding to the action at this timestep
                    for (int z = 0; z < TNS; ++z)
                    {
                        // calculate temp_vals
                        double top;
                        if (EM_USE_RO)
                        {
                            top = alpha[t][x]*est_t[x][cact][z]*est_o[z][cact][pomdp.obs[t]]*beta[t+1][z]*est_ro[z][pomdp.actions[t+1]][pomdp.reward_obs[t+1]];
                        }
                        else
                        {
                            top = alpha[t][x]*est_t[x][cact][z]*est_o[z][cact][pomdp.obs[t]]*beta[t+1][z];
                        }
                        sum += top;
                        temp_vals[x][z] = top;
                    }
                }
                
                // calculate xi
                // next normalize and add to the xi_sum
                for (int x = 0; x < TNS; ++x)
                {
                    for (int z = 0; z < TNS; ++z)
                    {
                        if (sum > TOOSMALL)
                        {
                            // subtract to divide to normalize
                            temp_vals[x][z] /= sum;
                            // acc it
                            xi_sum[x][cact][z] += temp_vals[x][z];
                        }
                        // otherwise don't add anything
                    }
                }
            }
            // now it's easy to compute the estimated probs using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 0; y < TNA; ++y)
                {
                    // compute the confidence intervals

                    double fake_count = 0;
                    if (scale_t == 1.0){
                        fake_count = gamma_action_sum[x][y];
                    }
                    else {
                        for (int c = 0; c < pomdp.actions.size(); c++)
                        {
                            if (y == pomdp.actions[c]) {fake_count += 1;}
                        }
                    }
                    double ci_radius = fake_count >= 1.0 ? sqrt((0.5/(scale_t * fake_count))*log (2.0/confidence_alpha)) : 1.0;
                    
                    // for normalizing the transitions
                    double sum = 0.0;
                    for (int z = 0; z < TNS; ++z)
                    {
                        if (gamma_action_sum_less[x][y] <= TOOSMALL)
                        {
                            // if no data, then uniform
                            est_t[x][y][z] = 1.0/TNS;
                        }
                        else
                        {
                            est_t[x][y][z] = xi_sum[x][y][z] / gamma_action_sum_less[x][y];
                        }
                        sum += est_t[x][y][z];
                        err_t[x][y][z] = ci_radius;
                        //err_t[x][y][z] = 1;
                    }
                    assert(sum >0);
                    // normalize the transitions
                    for (int z = 0; z < TNS; ++z)
                    {
                        est_t[x][y][z] /= sum;
                    }
                }
            }
        }
        else
        {
            // if not estimating the transitions, just set it to the true settings
            copy_matrix(pomdp.t, est_t);
            zero_out(err_t);
        }
        
        // estimate obs
        if (EST_O)
        {
            // add up all things for the obs
            for (int t = 1; t < T; ++t)
            {
                // previous action
                int pact = pomdp.actions[t-1];
                
                // current obs
                int cobs = pomdp.obs[t-1];
                
                // calculate gamma_action_sum
                for (int x = 0; x < TNS; ++x)
                {
                    // update only the entries corresponding to the action at this timestep
                    obs_sum[x][pact][cobs] += gamma[t][x];
                }
            }
            
            // now it's easy to compute the estimated probs using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 2; y < TNA; ++y)
                {
                    // compute the confidence intervals

                    double fake_count = 0;
                    if (scale_o == 1.0)
                    {
                        fake_count = gamma_action_sum[x][y];
                    }
                    else 
                    {
                        fake_count = numtriples[y - 2];
                    }

                    double ci_radius = fake_count >= 1 ? sqrt((0.5/(scale_o * fake_count))*log (2.0/confidence_alpha)) : 1.0;
                    
                    // for normalizing the obs
                    double sum = 0.0;
                    // compute the expected obs
                    for (int z = 0; z < TNO; ++z)
                    {
                        if (gamma_action_sum_prev[x][y] <= TOOSMALL)
                        {
                            // if no data, then uniform
                            est_o[x][y][z] = 1.0/TNO;
                        }
                        else
                        {
                            est_o[x][y][z] = obs_sum[x][y][z] / gamma_action_sum_prev[x][y];
                        }
                        sum += est_o[x][y][z];
                        err_o[x][y][z] = ci_radius;
                    }
                    // normalize the reward obs
                    for (int z = 0; z < TNO; ++z)
                    {
                        est_o[x][y][z] /= sum;
                    }
                }
            }
        }
        else
        {
            // use real obs
            copy_matrix(pomdp.o, est_o);
            zero_out(err_o);
        }
        
        // estimate reward obs
        if (EST_RO)
        {
            // add up all things for the reward obs
            for (int t = 0; t < T; ++t)
            {
                // current action
                int cact = pomdp.actions[t];
                
                // current reward obs
                int crobs = pomdp.reward_obs[t];
                
                // calculate gamma_action_sum
                for (int x = 0; x < TNS; ++x)
                {
                    // update only the entries corresponding to the action at this timestep
                    rho_sum[x][cact][crobs] += gamma[t][x];
                }
            }
            
            // now it's easy to compute the estimated probs using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 0; y < TNA; ++y)
                {
                    // compute the confidence intervals
                    
                    double fake_count = 0;
                    if (scale_ro == 1.0){
                        fake_count = gamma_action_sum[x][y];
                    }
                    else {
                        for (int c = 0; c < pomdp.actions.size(); c++)
                        {
                            if (y == pomdp.actions[c]) {fake_count += 1;}
                        }
                    }
                    double ci_radius = fake_count >= 1.0 ? sqrt((0.5/(scale_ro * fake_count))*log (2.0/confidence_alpha)) : 1.0;
                    
                    // for normalizing the reward obs
                    double sum = 0.0;
                    // compute the expected reward obs
                    for (int z = 0; z < TNR; ++z)
                    {
                        if (gamma_action_sum[x][y] <= TOOSMALL)
                        {
                            // if no data, then uniform
                            est_ro[x][y][z] = 1.0/TNR;
                        }
                        else
                        {
                            est_ro[x][y][z] = rho_sum[x][y][z] / gamma_action_sum[x][y];
                        }
                        sum += est_ro[x][y][z];
                        err_ro[x][y][z] = ci_radius;
                    }
                    // normalize the reward obs
                    for (int z = 0; z < TNR; ++z)
                    {
                        est_ro[x][y][z] /= sum;
                    }
                }
            }
        }
        else
        {
            // if not estimating, use the true params
            copy_matrix(pomdp.ro, est_ro);
            zero_out(err_ro);
        }
        
        // estimate the rewards
        if (EST_R)
        {
            // add up all things for the estimated rewards
            for (int t = 0; t < T; ++t)
            {
                // current action
                int cact = pomdp.actions[t];
                
                // calculate ex_reward
                for (int x = 0; x < TNS; ++x)
                {
                    // weighted by the belief prob
                    ex_reward[x][cact] += pomdp.rewards[t] * gamma[t][x];
                }
            }
            
            // now it's easy to compute the estimated rewards using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 0; y < TNA; ++y)
                {
                    double fake_count = 0;
                    if (scale_r == 1.0){
                        fake_count = gamma_action_sum[x][y];
                    }
                    else
                    {
                        for (int c = 0; c < pomdp.actions.size(); c++)
                        {
                            if (y == pomdp.actions[c]) {fake_count += 1;}
                        }
                    }
                    double ci_radius = fake_count >= 1.0 ? REWARD_GAP*sqrt((0.5/(scale_r * fake_count))*log (2.0/confidence_alpha)) : 1.0;
                    
                    // compute expected reward
                    if (gamma_action_sum[x][y] <= TOOSMALL)
                    {
                        // if no data, then rmax
                        opt_r[x][y] = pomdp.rmax;
                        est_r[x][y] = pomdp.rmax;
                    }
                    else
                    {
                        opt_r[x][y] = min(ex_reward[x][y] / gamma_action_sum[x][y] + ci_radius, pomdp.rmax);
                        est_r[x][y] = min(ex_reward[x][y] / gamma_action_sum[x][y], pomdp.rmax);
                    }
                }
            }
        }
        else
        {
            // use the true rewards
            copy_matrix(pomdp.r, est_r);
            copy_matrix(pomdp.r, opt_r);
        }

        // pomdp.reset_nonlisten_params(est_t, err_t, est_o, err_o);
        
        // for debugging
        if (0 and iters == num_iters-1)
        {
            cout << "printing out gamma action sum" << endl;
            print_matrix(gamma_action_sum);
        }
    }
    
    // for debugging
    if (0)
    {
        cout << "em called" << endl;
        for (int l = 0; l <= T; l++){
            cout << "sim step " << l << ", s " << pomdp.states[l] << ", r " << pomdp.rewards[l] << ", a " << pomdp.actions[l] << ", o " << pomdp.obs[l] << ": ";
            for (int i = 0; i < pomdp.numstates; i++) {
                cout << (gamma[l][i]) << " ";
            }
            cout << "| ";
            for (int i = 0; i < pomdp.numstates; i++) {
                cout << (alpha[l][i]) << " ";
            }
            cout << "| ";
            for (int i = 0; i < pomdp.numstates; i++) {
                cout << (beta[l][i]) << " ";
            }
            cout << endl;
        }
    }
    
    double log_const = log(1.0);
    
    // calculate the log likelihood by recomputing the alphas and keeping the constants around
    // initialize the base cases of alpha
    for (int i = 0; i < TNS; ++i)
    {
        alpha[0][i] = pi[i];
    }
    
    // recursively build up alpha and beta from previous values
    for (int t = 1; t < T; ++t)
    {
        // alpha goes forward and beta goes backward
        int ai = t;
        // to normalize alpha and beta
        double a_denom = 0;
        // do the recursive update
        for (int i = 0; i < TNS; ++i) {
            double asum = 0;
            for (int j = 0; j < TNS; ++j)
            {
                asum += alpha[ai-1][j]*est_t[j][pomdp.actions[ai-1]][i];
            }
            alpha[ai][i] = asum * est_o[i][pomdp.actions[ai-1]][pomdp.obs[ai-1]];
            
            a_denom += alpha[ai][i];
        }
        
        assert(a_denom > 0);
        
        // normalize alpha and add in the normalization constant
        for (int i = 0; i < TNS; ++i)
        {
            alpha[ai][i] /= a_denom;
        }
        
        log_const = mylogmul(log_const, log(a_denom));
    }
    
    // now calculate the log likelihood by summing up the last of alpha
    double ll = log(0.0);
    for (int i = 0; i < TNS; ++i)
    {
        ll = mylogadd(ll, mylogmul(log_const, log(alpha[T-1][i])));
    }
    
    return ll;
}

double likelihood(POMDP &pomdp, double est_o[TNS][TNA][TNO], double est_t[TNS][TNA][TNO])   
{
    const int T = pomdp.states.size();
    double pi[2] = {0.5, 0.5};
    if (T <= 0)
    {
        return 0;
    }
    
    double alpha[T][TNS];
    double log_const = log(1.0);
    
    // calculate the log likelihood by recomputing the alphas and keeping the constants around
    // initialize the base cases of alpha
    for (int i = 0; i < TNS; ++i)
    {
        alpha[0][i] = pi[i];
    }
    
    // recursively build up alpha and beta from previous values
    for (int t = 1; t < T; ++t)
    {
        // alpha goes forward and beta goes backward
        int ai = t;
        // to normalize alpha and beta
        double a_denom = 0;
        // do the recursive update
        for (int i = 0; i < TNS; ++i) {
            double asum = 0;
            for (int j = 0; j < TNS; ++j)
            {
                asum += alpha[ai-1][j]*est_t[j][pomdp.actions[ai-1]][i];
            }
            alpha[ai][i] = asum * est_o[i][pomdp.actions[ai-1]][pomdp.obs[ai-1]];
            
            a_denom += alpha[ai][i];
        }
        if (a_denom <= 0)
        {
            // cout << "a_denom: " << a_denom << endl;
            return - numeric_limits<double>::infinity();
        }
        assert(a_denom > 0);
        
        // normalize alpha and add in the normalization constant
        for (int i = 0; i < TNS; ++i)
        {
            alpha[ai][i] /= a_denom;
        }
        
        log_const = mylogmul(log_const, log(a_denom));
    }
    
    // now calculate the log likelihood by summing up the last of alpha
    double ll = log(0.0);
    for (int i = 0; i < TNS; ++i)
    {
        ll = mylogadd(ll, mylogmul(log_const, log(alpha[T-1][i])));
    }
    
    return ll;
}

void test_opt(bool use_opt, string const &rot_out, string const &cummreward_out, string const &actions_out, double decay = -1)
{
    // use for testing different CIs
    // unsigned initial_seed = 792486162;
    unsigned initial_seed = 983446855;

    // unsigned initial_seed = time(0);
    
    // generate a bunch of seeds, one for each rep
    sample_seed(initial_seed);

    vector<unsigned> seeds;
    // seed for bad mean estimates
    // seeds.push_back(798002262);
    // seed for bad optimistic estimates
    // seeds.push_back(3045999349);
    for (int i = 0; i < 20; ++i)
    {
        seeds.push_back(sample_rand());
    }
    //seeds.push_back(2546248239);
    
    int reps = seeds.size();
    int expeps = 1;
    int eps = expeps + 0;
    int max_steps_per_ep = 40000;
    int steps = max_steps_per_ep * eps;
    double sum_rewards = 0;
    double prev_sum = 0;
    int nrandtriples = 0;
    double fractwos = 0;
    double fracthrees = 0;
    vector<double> policy;

    double timeavg = 0;

    ofstream histogram_r(cummreward_out);
    ofstream output_a(actions_out);

    vector<double> eprs(eps, 0.0);
    vector<double> rs(steps, 0.0);

    mat I = eye<mat>(TNO, TNO);
    
    double est_t[TNS][TNA][TNS];
    double err_t[TNS][TNA][TNS];
    
    double est_o[TNS][TNA][TNO];
    double err_o[TNS][TNA][TNO];
    
    double est_ro[TNS][TNA][TNR];
    double err_ro[TNS][TNA][TNR];
    
    double est_r[TNS][TNA];
    double opt_r[TNS][TNA];

    // double scale_t = 5.0;
    // double scale_o = 5.0;
    // double scale_ro = 5.0;
    // double scale_r = 5.0;
    double scale_t = 1.0;
    double scale_o = 1.0;
    double scale_ro = 1.0;
    double scale_r = 1.0;

    for (int rep = 0; rep < reps; ++rep)
    {
        // scale -= 0.01;
        // keep track of episodes
        int curr_ep = 0;
        int curr_ep_step = 0;
        int last_triple = -1;
        int momcount[TNA];
        int prev_action = -1;
        double epirs = 0.0;

        for (int a = 0; a < TNA; a++)
        {
            momcount[a] = 0;
        }

        vector<cube> obs;
        cube newobs;
        for (int i = 0; i < TNA; i++)
        {
            cube tmp;
            obs.push_back(tmp);
        }

        // keep track of whether the model was updated
        bool model_updated = true;
        bool update_model = true;
        // bool mom_updated[4] = {false, false, false, false};

        int no_triples = 0;
    
        unsigned seed = seeds.at(rep);
        sample_seed(seed);
        cout << "---- Start rep " << rep << endl;
        cout << "seed " << seed << endl;
        rept = clock() - clock();
        // cout << rep << endl;
        // initialize parameters
        POMDP pomdp;
        for (int x = 0; x < TNS; x++) {
            for (int y = 0; y < TNA; y++) {
                est_r[x][y] = BIG_REWARD;
                opt_r[x][y] = BIG_REWARD;
                for (int z = 0; z < TNS; z++) {
                    est_t[x][y][z] = 0;
                    err_t[x][y][z] = 1;
                }
                for (int z = 0; z < TNO; z++) {
                    est_o[x][y][z] = 0;
                    err_o[x][y][z] = 0;
                }
                for (int z = 0; z < TNR; z++) {
                    est_ro[x][y][z] = 0;
                    err_ro[x][y][z] = 1;
                }
            }
        }
        // for (int i = 0; i < 1000; i++)
        // {
        //     int a;
        //     // double randunif = sample_unif();
        //     // if (randunif <= 0.5)
        //     // {
        //     //     a = 1;
        //     // }
        //     // else 
        //     // {
        //     //     a = 0;
        //     // }
        //     // a = sample_int(0, 1);
        //     for (int j = 0; j < 3; j++)
        //     {
        //         pomdp.step(0);
        //     }
        //     newobs = randu<cube>(TNO, obs.at(a).n_cols + 1, 3);
        //     if (newobs.n_cols > 1)
        //     { 
        //         newobs.subcube(0, 0, 0, TNO - 1, obs.at(a).n_cols - 1, 2) = obs.at(a);
        //     }
        //     for (int k = 0; k < 3; k++)
        //     {
        //         newobs.slice(k).col(obs.at(a).n_cols) = I.col(pomdp.obs[i*3 + k]);
        //     }
        //     obs.at(a) = newobs;
        // }
        for (int i = 0; i < nrandtriples; i++)
        {
            cout << i << endl;
            int a;
            int a2;
            // double randunif = sample_unif();
            // if (randunif <= 0.5)
            // {
            //     a = 1;
            // }
            // else 
            // {
            //     a = 0;
            // }
            // pomdp.new_episode();
            // a = sample_int(2, 3);
            for (int j = 0; j < 3; j++)
            {
                a = sample_int(0, 3);
                if (j == 1) 
                {
                    a = sample_int(2, 3);
                    a2 = a;
                }
                pomdp.step(a);
            }
            pomdp.new_episode();
            newobs = randu<cube>(TNO, obs.at(a2).n_cols + 1, 3);
            if (newobs.n_cols > 1)
            { 
                newobs.subcube(0, 0, 0, TNO - 1, obs.at(a2).n_cols - 1, 2) = obs.at(a2);
            }
            for (int k = 0; k < 3; k++)
            {
                newobs.slice(k).col(obs.at(a2).n_cols) = I.col(pomdp.obs[(i*3) + k]);
            }
            obs.at(a2) = newobs;
            // a = sample_int(0, 1);
            // pomdp.step(a);
            // cout << newobs.n_cols << endl;
            // cout << obs.at(a).n_cols << endl;
        }

        // for (int i = 0; i < 10000; i++)
        // {
        //     int a;
        //     double randunif = sample_unif();
        //     if (randunif <= 0.5)
        //     {
        //         a = 2;
        //     }
        //     else 
        //     {
        //         a = 3;
        //     }
        //     a = sample_int(0, 1);
        //     pomdp.step(a);
        // }
        initialize(pomdp, est_t, est_o, est_ro);
        int a;
        Planning<POMDP,double[TNS][TNA][TNS],double[TNS][TNA][TNO],double[TNS][TNA]> plan(pomdp);
        for (int iter = 0; iter < steps; iter++) {
            ++curr_ep_step;
            //cout << "---------- Iteration " << iter+1 << " ----------" << endl;
            //cout << "Curr Belief -- ";
            //print_vector(plan.curr_belief);
            // t = clock();
            // if nonoptimistic
            // cout << obs.at(2).n_cols << endl;
            // cout << obs.at(3).n_cols << endl;
            if (not use_opt or curr_ep >= expeps) {
                // zero out the confidence intervals
                zero_out(err_t);
                zero_out(err_o);
                // set opt r to be same as est r
                copy_matrix(est_r, opt_r);
            }

            int next_action = -1;

            // ######################### 
            // Use this when we want to model the optimal policy.
            // if (iter == 0) {
            //     next_action = plan.backup_plan(pomdp.t, err_t, pomdp.o, err_o, pomdp.r, true, 100);
            // }
            // else
            // {
            //     next_action = plan.backup_plan(pomdp.t, err_t, pomdp.o, err_o, pomdp.r, false, 1);
            // }
            // ##########################
            // Do this if we want to enforce the first three actions to be the same.  We want to avoid this if possible.
            // If you uncomment this if statement, you need an else if for the next statement.
            // if ((curr_ep_step == 2 or curr_ep_step == 3) and curr_ep < expeps) {
            //     next_action = prev_action;
            // }
            // ##########################

            if (model_updated)
            {
                clock_t starttime = clock();
                next_action = plan.backup_plan(est_t, err_t, est_o, err_o, opt_r, true, 40);
                rept += clock() - starttime;
                model_updated = false;
                // cout << "estimated obs" << endl;
                // print_both(est_o, err_o);
                // mom_updated[2] = false;
                // mom_updated[3] = false;
                //plan.print_points();
                //cout << "opt t" << endl;
                //print_matrix(plan.opt_t);
                //cout << "opt z" << endl;
                //print_matrix(plan.opt_z);
            }
            else
            {
                next_action = plan.backup_plan(est_t, err_t, est_o, err_o, opt_r, false, 1);
            }

            // if (iter % 4 == 0)
            // {
            //     a = sample_int(2, 3);
            //     next_action = sample_int(0, 1);
            //     // if (iter >= 4)
            //     // {
            //     //     cout << pomdp.actions[iter - 3] << " " << pomdp.actions[iter - 2] << " " << pomdp.actions[iter - 1] << " " << next_action << endl;
            //     // }
            // }
            // else 
            // {
            //     next_action = a;
            // }
            // egreedy random action with decay
            // but stop egreedy for the last 100 steps
            if (decay > 0.0 and iter + 100 < steps)
            {
                double ep_chance = 1.0 / (iter/decay + 1.0);
                if (sample_unif() < ep_chance)
                {
                    // egreedy to do actions
                    next_action = sample_int(0, pomdp.numactions-1);
                }
            }
            assert (next_action >= 0 and next_action < TNA);
            
            // advance the pomdp
            pomdp.step(next_action);
            prev_action = next_action;
            //cout << "Curr Belief -- ";
            //print_vector(plan.curr_belief);
            // if ((pomdp.actions[iter] == 2 or pomdp.actions[iter] == 3) and (pomdp.actions[iter-1] == 2 or pomdp.actions[iter-1] == 3) and (pomdp.actions[iter-2] == 2 or pomdp.actions[iter-2] == 3) and last_triple < iter - 2 and curr_ep_step == 3) 
            if (curr_ep_step >= 3 and (pomdp.actions[iter-1] >= 2) and last_triple < iter - 2)
            {
                int a = pomdp.actions[iter-1];
                newobs = randu<cube>(TNO, obs.at(a).n_cols + 1, 3);
                if (newobs.n_cols > 1)
                { 
                    newobs.subcube(0, 0, 0, TNO - 1, obs.at(a).n_cols - 1, 2) = obs.at(a);
                }
                for (int i = 0; i < 3; i++)
                {
                    newobs.slice(i).col(obs.at(a).n_cols) = I.col(pomdp.obs[iter-2+i]);
                }
                obs.at(a) = newobs;
                last_triple = iter;
                no_triples = 0;
                update_model = true;
            }
            else
            {
                update_model == false;
                no_triples += 1;
                if (no_triples >= 50)
                {
                    cout << "NO SAMPLES FOR AT LEAST 50 STEPS." << endl;
                }
            }
            // cout << "Num triples for bad sensor: " << obs.at(2).n_cols << endl;
            // cout << "Num triples for good sensor: " << obs.at(3).n_cols << endl;
            // if (pomdp.actions[iter] == pomdp.actions[iter - 1] and last_triple < iter - 2 and curr_ep_step == 2) 
            // {
            //     int a = pomdp.actions[iter];
            //     newobs = randu<cube>(TNO, obs.at(a).n_cols + 1, 3);
            //     if (newobs.n_cols > 1)
            //     { 
            //         newobs.subcube(0, 0, 0, TNO - 1, obs.at(a).n_cols - 1, 2) = obs.at(a);
            //     }
            //     for (int i = 0; i < 3; i++)
            //     {
            //         newobs.slice(i).col(obs.at(a).n_cols) = I.col(pomdp.obs[iter-2+i]);
            //     }
            //     obs.at(a) = newobs;
            //     last_triple = iter;
            // }
            // for (int i = 0; i < TNA; i++)
            // {
            //     cout << "Samples for action " << i << ": " << obs.at(i).n_cols << endl;
            // }

            // debug information
            // if (iter % 999 == 0)
            if (0)
            {
                cout << "-------------------- Iteration " << iter << " --------------------" << endl;
                // cout << "(s,a,r) = " << pomdp.states.back() << "," << pomdp.actions.back() << "," << pomdp.rewards.back() << endl;
                // cout << "Curr Belief -- ";
                // print_vector(plan.curr_belief);
                // cout << "estimated transitions" << endl;
                // print_both(est_t, err_t);
                // cout << "optimistic transitions" << endl;
                // print_matrix(plan.opt_t);
                cout << "estimated obs" << endl;
                print_both(est_o, err_o);
                cout << "optimistic obs" << endl;
                print_matrix(plan.opt_z);
                // cout << "estimated rewards" << endl;
                // print_both(est_r, opt_r);
                // cout << "-------------------- Iteration " << iter+1 << " --------------------" << endl;
            }
            
            // update beliefs for next step
            plan.belief_update_full();

            int initial_burn_in = 0;
            // if (curr_ep_step == 1 and curr_ep < expeps)
            // if (update_model and curr_ep < expeps)
            // if ((curr_ep == initial_burn_in or (curr_ep > initial_burn_in and (curr_ep-initial_burn_in) % 50  == 0)) and curr_ep_step == 1)            
            // if (0)
            if (iter % 50 == 0)
            {
                initialize(pomdp, est_t, est_o, est_ro);
                // pomdp.reset_nonlisten_params(est_t, err_t, est_o, err_o);
                int numtriples[TNA - 2];
                for (int a = 2; a < TNA; a++)
                {
                    numtriples[a - 2] = obs.at(a).n_cols;
                }

                em(pomdp, est_t, err_t, est_o, err_o, est_ro, err_ro, est_r, opt_r, numtriples, scale_t, scale_o, scale_ro, scale_r);

                for (int a = 2; a < TNA; a++)
                {
                    if (est_o[0][a][0] <= 0.5 or est_o[1][a][1] <= 0.5)
                    {
                        double temp_o[TNS][TNO];
                        double temp_t[TNS][TNS];
                        // print_matrix(est_o);
                        for (int i = 0; i < TNS; i++)
                        {
                            for (int k = 0; k < TNO; k++)
                            {
                                temp_o[i][k] = est_o[(i + 1) % 2][a][k];
                            }
                        }
                        for (int i = 0; i < TNS; i++)
                        {
                            for (int k = 0; k < TNO; k++)
                            {
                                est_o[i][a][k] = temp_o[i][k];
                            }
                        }
                    }
                }
                model_updated = true;
                // cout << "model updated" << endl;
                // cout << "esimated transitions" << endl;
                // print_both(est_t, err_t);
                // cout << "estimated obs" << endl;
                // print_both(est_o, err_o);
                // cout << "esimated reward obs" << endl;
                // print_both(est_ro, err_ro);
                // cout << "estimated rewards" << endl;
                // print_both(est_r, opt_r);
            }
            // if ((curr_ep % 50 == 0) and (curr_ep_step == 0))
            // {
            //     cout << "Curr episode: " << curr_ep << endl;
            //     print_matrix(est_o);
            // }
            if (0)
            // if (curr_ep_step == 1 and curr_ep < expeps)
            // if (update_model and curr_ep < expeps)
            // if ((curr_ep == initial_burn_in or (curr_ep > initial_burn_in and (curr_ep-initial_burn_in) % 50  == 0)) and curr_ep_step == 1)
            {
                initialize(pomdp, est_t, est_o, est_ro);
                double new_o[TNS][TNA][TNO];
                double best_o[TNS][TNA][TNO];

                for (int i = 2; i < TNA; i++)
                {
                    mat o_mat = zeros<mat>(TNS, TNO);
                    mat t_mat = zeros<mat>(TNS, TNS);
                    if (obs.at(i).n_cols > 0)
                    {
                        LearnHMM(o_mat, t_mat, obs.at(i), TNS);
                        urowvec nonzeros = any(o_mat, 0);
                        if (all(nonzeros))
                        {
                            // cout << o_mat << endl;
                            // cout << obs.at(i).n_cols << endl;
                            momcount[i] += 1;
                            for (int j = 0; j < TNS; j++)
                            {                 
                                for (int k = 0; k < TNO; k++)
                                {
                                    est_o[j][i][k] = o_mat(k, j);
                                }
                            }
                            // mom_updated[i] = true;
                        }
                    }
                }
                for (int a = 2; a < TNA; a++)
                {
                    if (est_o[0][a][0] <= 0.5 or est_o[1][a][1] <= 0.5)
                    {
                        double temp_o[TNS][TNO];
                        double temp_t[TNS][TNS];
                        // print_matrix(est_o);
                        for (int i = 0; i < TNS; i++)
                        {
                            for (int k = 0; k < TNO; k++)
                            {
                                temp_o[i][k] = est_o[(i + 1) % 2][a][k];
                            }
                        }
                        for (int i = 0; i < TNS; i++)
                        {
                            for (int k = 0; k < TNO; k++)
                            {
                                est_o[i][a][k] = temp_o[i][k];
                            }
                        }
                    }
                }
            
                // cout << "MoM ll: " << likelihood(pomdp, est_o, est_t) << endl;
                int statearray[TNA][TNS];
                for (int i = 0; i < TNS; i++)
                {
                    statearray[0][i] = i;
                    statearray[1][i] = i;
                }
                copy_matrix(est_o, best_o);
                do
                {
                    do
                    {
                        for (int i = 0; i < TNS; i++)
                        {
                            for (int j = 0; j < TNA; j++)
                            {                    
                                for (int k = 0; k < TNO; k++)
                                {
                                    // if (j == 0 and i == 1)
                                    // {
                                    //     new_o[i][j][k] = est_o[i][j][statearray[0][k]];
                                    // }
                                    if (j < 2)
                                    {
                                        new_o[i][j][k] = est_o[i][j][k];
                                    }
                                    else if (j == 2)
                                    {
                                        new_o[i][j][k] = est_o[statearray[0][i]][j][k]; 
                                    }
                                    else
                                    {
                                        new_o[i][j][k] = est_o[statearray[1][i]][j][k]; 
                                    }
                                }
                            }
                        }
                        for (int a = 2; a < TNA; a++)
                        {
                            if (new_o[0][a][0] <= 0.5 or est_o[1][a][1] <= 0.5)
                            {
                                double temp_o[TNS][TNA][TNO];
                                double temp_t[TNS][TNA][TNO];
                                // print_matrix(est_o);
                                for (int i = 0; i < TNS; i++)
                                {
                                    for (int k = 0; k < TNO; k++)
                                    {
                                        temp_o[i][a][k] = new_o[(i + 1) % 2][a][k];
                                    }
                                }
                                for (int i = 0; i < TNS; i++)
                                {
                                    for (int k = 0; k < TNO; k++)
                                    {
                                        new_o[i][a][k] = temp_o[i][a][k];
                                    }
                                }
                            }
                        }
                        if (likelihood(pomdp, new_o, est_t) > likelihood(pomdp, best_o, est_t))
                        {
                            copy_matrix(new_o, best_o);
                        }
                    } while (next_permutation(statearray[1], statearray[1] + TNS));        
                } while (next_permutation(statearray[0], statearray[0] + TNS));
                copy_matrix(best_o, est_o);
            }
            // if ((curr_ep_step == 1) and curr_ep >= expeps)
            // {
            //     cout << "Curr episode: " << curr_ep << endl;
            //     print_both(est_o, err_o);
            // }
            double recent_reward = pomdp.rewards.back();
            epirs += recent_reward;
            rs[iter] += recent_reward;
            // if (iter % 4 == 0)
            // {
            //     ++curr_ep;
            // }
            if (curr_ep_step >= max_steps_per_ep)
            // if (abs(recent_reward - 10) < 0.01 or abs(recent_reward+100) < 0.01 or curr_ep_step >= max_steps_per_ep)
            {
                // end of an episode
                eprs[curr_ep] += epirs/(curr_ep_step);
                epirs = 0;
                ++curr_ep;
                curr_ep_step = 0;
                pomdp.new_episode();
                plan.reset_curr_belief();
                // cout << obs.at(2).n_cols << endl;
                // cout << obs.at(3).n_cols << endl;
            }
            if (curr_ep >= eps)
            {
                break;
            }
        }
        // cout << obs.at(2).n_cols << " " << obs.at(3).n_cols << endl;
        for (size_t i = 0; i < pomdp.rewards.size(); ++i)
        {
            //rs[i] += pomdp.rewards[i];
            sum_rewards += pomdp.rewards[i];
        }
        histogram_r << sum_rewards - prev_sum << endl;
        prev_sum = sum_rewards;
        // show some stats on the run
        if (1)
        {
            // cout << "states" << endl;
            // for (size_t i = 0; i < pomdp.rewards.size(); ++i)
            // {
            //     cout << pomdp.states[i] << " ";
            // }
            // cout << endl;
            cout << "obs" << endl;
            for (size_t i = nrandtriples * 3; i < pomdp.obs.size(); ++i)
            {
                cout << pomdp.obs[i] << " ";
            }
            cout << endl;
            cout << "actions" << endl;
            output_a << "actions" << endl;
            bool check = true;
            int num = 0;
            int k = 0;
            int twos = 0;
            int threes = 0;
            for (size_t i = nrandtriples * 3; i < pomdp.actions.size(); ++i)
            {
                cout << pomdp.actions[i] << " ";
                output_a << pomdp.actions[i] << " ";
                if (pomdp.actions[i] == 2 and k >= expeps)
                {
                    twos += 1;
                }
                if (pomdp.actions[i] == 3 and k >= expeps)
                {
                    threes += 1;
                }
                if (not ((-3 < num < 3 and pomdp.actions[i] == 3) or (num == 3 and pomdp.actions[i] == 1) or (num == -3 and pomdp.actions[i] == 0)))
                {
                    check = false;
                }

                if (pomdp.actions[i] == 0 or pomdp.actions[i] == 1)
                {
                    num = 0;
                    if (policy.size() < k + 1)
                    {
                        policy.push_back(0);
                    }
                    if (check) 
                    {
                        policy[k] += 1;
                    }
                    check = true;
                    k += 1;
                }
                else if (pomdp.obs[i] == 0) 
                {
                    num -= 1;
                }
                else
                {
                    num += 1;
                }
            }
            if (twos >= eps - expeps)
            {
                fractwos += 1.0;
            }
            if (threes >= eps - expeps)
            {
                fracthrees += 1.0;
            }
            cout << endl;
            output_a << endl;
            // cout << "reward obs" << endl;
            // for (size_t i = 0; i < pomdp.reward_obs.size(); ++i)
            // {
            //     cout << pomdp.reward_obs[i] << " ";
            // }
            // cout << endl;
            //cout << "rewards" << endl;
            //for (size_t i = 0; i < pomdp.rewards.size(); ++i)
            //{
                //cout << pomdp.rewards[i] << " ";
            //}
            //cout << endl;
        }
        // cout << "MoM count: 2: " << momcount[2] << ", 3: " << momcount[3] << endl;

        // cout << "Rewards: " << sum_rewards - prev_sum << endl;
        cout << "seed " << seed << endl;
        // rept = clock() - rept;
        cout << "---- End Rep: " << ((float) rept)/CLOCKS_PER_SEC << endl << endl;
        timeavg += ((float) rept)/CLOCKS_PER_SEC;
    }
    cout << "Average time: " << timeavg/reps << endl;
    ofstream output_r(rot_out);
    // for (size_t i = 0; i < eprs.size(); ++i)
    // {
    //     eprs[i] /= reps;
    //     output_r << eprs[i] << endl;
    // }
    for (size_t i = 0; i < rs.size(); ++i)
    {
        rs[i] /= reps;
        output_r << rs[i] << endl;
    }

    // for (size_t i = 0; i < rs.size(); ++i)
    // {
    //     if (rs[i] == 0)
    //     {
    //         break;
    //     }
    //     rs[i] /= reps;
    //     output_r << rs[i] << endl;
    // }
    output_r << sum_rewards/reps << endl;
    histogram_r.close();
    output_r.close();

    double avgp = 0;
    for (size_t i = 0; i < policy.size(); ++i)
    {
        cout << fixed << setprecision(2) << policy[i]/reps << " ";
        if (i >= expeps)
        {
            avgp += policy[i]/reps;
        }
    }
    cout << "avg policy: " << avgp/(eps - expeps) << endl;
    cout << endl;
    cout << "Frac twos: " << fractwos/reps << endl;
    cout << "Frac threes: " << fracthrees/reps << endl;
    
    if (1)
    {
        //cout << "avg rewards" << setw(4) << endl;
        //for (size_t i = 0; i < rs.size(); ++i)
        //{
            //cout << setw(4) << rs[i] << " ";
        //}
        //cout << endl;
        double avgr = 0;
        cout << "avg reward per ep" << endl;
        for (size_t i = 0; i < eprs.size(); ++i)
        {
            // cout << fixed << setprecision(2) << setw(8) << eprs[i];
            if (i >= expeps)
            {
                avgr += eprs[i];
            }
        }
        // cout << endl;
        cout << "avg reward: " << avgr/(eps - expeps) << endl;
    }
    
    cout << "Cumulative reward " << sum_rewards/reps << endl;
}

int main()
{
    //test_random();
    
    //find_planning_params();
    
    //test_em("l2_out.txt", "l2_out_err.txt", "linf_out.txt", "linf_out_err.txt");
    
    // test_opt(true, "locmstiger_optimistic_rot_vlargeCI_250.txt", "locmstiger_optimistic_cumrewards_vlargeCI_250.txt", "locmstiger_optimistic_actions_vlargeCI_250.txt");
    // test_opt(false, "locmstiger_mean_rot_300_PT.txt", "locmstiger_mean_cummrewards_300_PT.txt", "locmstiger_mean_actions_300_PT.txt");
    // test_opt(true, "locmstiger_opt_rot_300_PT.txt", "locmstiger_opt_cummrewards_300_PT.txt", "locmstiger_opt_actions_300_PT.txt");

    // test_opt(true, "tiger_opt_fullrank_50s_9t.txt", "tiger_opt_fullrank_50s_cumrewards_9t.txt", "tiger_opt_fullrank_50s_actions_9t.txt");
    // test_opt(false, "tiger_mean_fullrank_50s_9t.txt", "tiger_mean_fullrank_50s_cumrewards_9t.txt", "tiger_mean_fullrank_50s_actions_9t.txt");
    // test_opt(true, "tiger_em_opt_fullrank_6t_2listens.txt", "tiger_opt_fullrank_cumrewards_6t_2listens.txt", "tiger_opt_fullrank_actions_6t_2listens.txt");
    test_opt(false, "tiger_em_mean_fullrank_6t_2listens.txt", "tiger_mean_fullrank_cumrewards_6t_2listens.txt", "tiger_mean_fullrank_actions_6t_2listens.txt");
    // test_opt(true, "tiger_em_opt_fullrank_9t_4listens_100backups.txt", "tiger_opt_fullrank_cumrewards_9t_4listens_100backups.txt", "tiger_opt_fullrank_actions_9t_4listens_100backups.txt");
    // test_opt(true, "tiger_em_opt_fullrank_9t_4listens_40beliefs.txt", "tiger_opt_fullrank_cumrewards_9t_4listens_40beliefs.txt", "tiger_opt_fullrank_actions_9t_4listens_40beliefs.txt");

    // test_opt(true, "tiger_em_opt_rewards.txt", "tiger_opt_fullrank_cumrewards.txt", "tiger_opt_fullrank_actions.txt");
    // test_opt(false, "tiger_em_mean_rewards.txt", "tiger_opt_fullrank_cumrewards.txt", "tiger_opt_fullrank_actions.txt");

    // test_opt(false, "newopt90.txt", "newopt2.txt", "newopt3.txt");
    // test_opt(false, "newmean.txt", "newmean2.txt", "newmean3.txt");
    // test_opt(true, "jargon.txt", "jargon1.txt", "jargon2.txt");
    // test_opt(false, "optimal_9t.txt", "jargon1.txt", "jargon2.txt");

    //int fs[] = {300};
    //int fs[] = {40,60,80,100};
    //int fs[] = {60,70,80,90,100};
    //for (int f: fs)
    //{
        //ostringstream outname;
        //outname << "mean_rewards_f" << f << ".txt";
        //test_opt(false, outname.str(), f);
    //}
    
    // let's optimizie cumulative reward over the egreedy param for 20 points and 10 reps and 300 steps
    // it looks like 10 can get to 0.55 to 0.60
    // it looks like 50 can get to 0.55 to 0.65
    // it looks like 100 is just terrible and gets like 0.35
    // it looks like 300 is just terrible and gets like 0.35
    
    
    return 0;
}
