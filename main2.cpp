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
#define EM_USE_RO true
#define EST_T true
#define EST_O true
#define EST_RO (EM_USE_RO and true)
#define EST_R true

#if 0
#define POMDP POMDP_OneArm
#define TIGER_NUMACTIONS 2
#define TIGER_NUMSTATES 2
#define TIGER_NUMOBS 2
#define TNR 3
#define START_STATE (0)
#define SMALL_REWARD 0.5
#define BIG_REWARD 1.0
#define SUCCESS_PROB 0.1
#define OBS_SUCCESS 0.9
#define ACC1 0.85
#define ACC2 0.9
#define LEARNING_EASE 0.3
#define REWARD_GAP 1.0
#else
#define POMDP POMDP_MSTiger
#define TIGER_NUMACTIONS 4
#define TIGER_NUMSTATES 2
#define TIGER_NUMOBS 2
#define TNR 3
#define START_STATE (0)
#define BIG_REWARD 10
#define SMALL_REWARD -100.00 // not used
#define SUCCESS_PROB 0.1 // not used
#define OBS_SUCCESS 0.85
#define ACC1 0.85
#define ACC2 0.9
#define LEARNING_EASE 0.3
#define REWARD_GAP 110.0
#endif

#include "planning.hpp"
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

double ACC[8] = {0.85, 0.9, 0.84, 0.9, 0.84, 0.86, 0.9, 0.91};

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

double l2_dist_squared(double const (&a1)[TNS][TNA][TNS], double const (&a2)[TNS][TNA][TNS])
{
    double d = 0.0;
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            for (int z = 0; z < TNS; ++z)
            {
                double diff = a1[x][y][z] - a2[x][y][z];
                d += diff*diff;
            }
        }
    }
    return d;
}

double l2_dist_squared(double const (&a1)[TNS][TNA][TNR], double const (&a2)[TNS][TNA][TNR])
{
    double d = 0.0;
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            for (int z = 0; z < TNR; ++z)
            {
                double diff = a1[x][y][z] - a2[x][y][z];
                d += diff*diff;
            }
        }
    }
    return d;
}

double l2_dist_squared(double const (&a1)[TNS][TNA], double const (&a2)[TNS][TNA])
{
    double d = 0.0;
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            double diff = a1[x][y] - a2[x][y];
            d += diff*diff;
        }
    }
    return d;
}

double linf_dist(double const (&a1)[TNS][TNA][TNS], double const (&a2)[TNS][TNA][TNS])
{
    double d = 0.0;
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            for (int z = 0; z < TNS; ++z)
            {
                double diff = a1[x][y][z] - a2[x][y][z];
                d = fmax(d,fabs(diff));
            }
        }
    }
    return d;
}

double linf_dist(double const (&a1)[TNS][TNA][TNR], double const (&a2)[TNS][TNA][TNR])
{
    double d = 0.0;
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            for (int z = 0; z < TNR; ++z)
            {
                double diff = a1[x][y][z] - a2[x][y][z];
                d = fmax(d,fabs(diff));
            }
        }
    }
    return d;
}

double linf_dist(double const (&a1)[TNS][TNA], double const (&a2)[TNS][TNA])
{
    double d = 0.0;
    for (int x = 0; x < TNS; ++x)
    {
        for (int y = 0; y < TNA; ++y)
        {
            double diff = a1[x][y] - a2[x][y];
            d = fmax(d,fabs(diff));
        }
    }
    return d;
}

struct POMDP_Tiger
{
    int numstates;
    int numactions;
    int numobs;

    int epi = 0;

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

    vector<vector <int>> actions;
    vector<vector <int>> obs;
    vector<vector <double>> rewards;
    
    // keep track of reward obs
    vector<vector <int>> reward_obs;
    
    // useful for debugging
    vector<vector <int>> states;

    int curr_state;

    POMDP_Tiger()
        : numstates(TNS), numactions(TNA), numobs(TNO),
          gamma(0.99), rmax(BIG_REWARD)
    {
        // actions are: go left or go right
        // states are: left, right
        // obs are: hear left, hear right
        // obs are fixed

        // the only rewards are going left in left and going right in right
        r[0][0] = -1;
        r[0][1] = 10;
        r[0][2] = -100;

        r[1][0] = -1;
        r[1][1] = -100;
        r[1][2] = 10.0;
        
        ro_map[0] = -1;
        ro_map[1] = -100;
        ro_map[2] = 10;
        
        // reflected in the reward obs
        ro[0][0][0] = 1.0; // listening gives -1 
        ro[0][0][1] = 0.0;
        ro[0][0][2] = 0.0;
        
        ro[0][1][0] = 0.0;
        ro[0][1][1] = 0.0;
        ro[0][1][2] = 1.0; // correctly opening the door
        
        ro[0][2][0] = 0.0;
        ro[0][2][1] = 1.0; // wrongly opening the door
        ro[0][2][2] = 0.0;
        
        ro[1][0][0] = 1.0; // listening gives -1
        ro[1][0][1] = 0.0;
        ro[1][0][2] = 0.0;
        
        ro[1][1][0] = 0.0;
        ro[1][1][1] = 1.0; // wrongly opening the door
        ro[1][1][2] = 0.0;
        
        ro[1][2][0] = 0.0;
        ro[1][2][1] = 0.0;
        ro[1][2][2] = 1.0; // correctly opening the door
        
        t[0][0][0] = 1.0; // stay in the same state
        t[0][0][1] = 0.0;
        
        t[0][1][0] = 0.5; // opening door resets
        t[0][1][1] = 0.5;
        
        t[0][2][0] = 0.5; // opening door resets
        t[0][2][1] = 0.5;
        
        t[1][0][0] = 0.0; // listening stays
        t[1][0][1] = 1.0;
        
        t[1][1][0] = 0.5; // open door resets
        t[1][1][1] = 0.5;
        
        t[1][2][0] = 0.5; // open door resets
        t[1][2][1] = 0.5;
        
        // listening gives mostly correct
        o[0][0][0] = OBS_SUCCESS;
        o[0][0][1] = 1.0 - OBS_SUCCESS;
        
        o[0][1][0] = 0.5;
        o[0][1][1] = 0.5;
        
        o[0][2][0] = 0.5;
        o[0][2][1] = 0.5;
        
        o[1][0][0] = 1.0 - OBS_SUCCESS;
        o[1][0][1] = OBS_SUCCESS;
        
        o[1][1][0] = 0.5;
        o[1][1][1] = 0.5;
        
        o[1][2][0] = 0.5;
        o[1][2][1] = 0.5;

        // start
        curr_state = sample_int(0, TNS-1);

        vector<int> epi_actions;
        vector<int> epi_obs;
        vector<double> epi_rewards;
        vector<int> epi_reward_obs;
        vector<int> epi_states;        

        actions.push_back(epi_actions);
        obs.push_back(epi_obs);
        rewards.push_back(epi_rewards);
        reward_obs.push_back(epi_reward_obs);
        states.push_back(epi_states);
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
        actions[epi].push_back(action);
        obs[epi].push_back(new_obs);
        reward_obs[epi].push_back(new_reward_obs);
        rewards[epi].push_back(ro_map[new_reward_obs]);
        states[epi].push_back(curr_state);
        curr_state = next_state;

        //cout << "action " << action << " " << prev_state << " -> " << next_state << endl;
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

struct POMDP_MSTiger
{
    int numstates;
    int numactions;
    int numobs;

    int epi = 0;

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

    vector<vector <int>> actions;
    vector<vector <int>> obs;
    vector<vector <double>> rewards;
    
    // keep track of reward obs
    vector<vector <int>> reward_obs;
    
    // useful for debugging
    vector<vector <int>> states;

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

        // standard version
        t[0][0][0] = 0.5; // opening door resets
        t[0][0][1] = 0.5;
        
        t[0][1][0] = 0.5; // opening door resets
        t[0][1][1] = 0.5;
        
        t[1][0][0] = 0.5; // opening door resets
        t[1][0][1] = 0.5;
        
        t[1][1][0] = 0.5; // opening door resets
        t[1][1][1] = 0.5;

        // full rank version (for MoM and comparing to MoM)
        // t[0][0][0] = 0.6; // opening door resets
        // t[0][0][1] = 0.4;
        
        // t[0][1][0] = 0.4; // opening door resets
        // t[0][1][1] = 0.6;
        
        // t[1][0][0] = 0.4; // opening door resets
        // t[1][0][1] = 0.6;
        
        // t[1][1][0] = 0.6; // opening door resets
        // t[1][1][1] = 0.4; 

        for (int i = 2; i < TNA; i++)
        {
            t[0][i][0] = 1.0; // listening stays
            t[0][i][1] = 0.0;

            t[1][i][0] = 0.0; // listening stays
            t[1][i][1] = 1.0;
        }


        // standard version
        o[0][0][0] = 0.5;
        o[0][0][1] = 0.5;
        
        o[0][1][0] = 0.5;
        o[0][1][1] = 0.5;
        
        o[1][0][0] = 0.5;
        o[1][0][1] = 0.5;
        
        o[1][1][0] = 0.5;
        o[1][1][1] = 0.5;

        // full rank version (for MoM and comparing to MoM)
        // o[0][0][0] = 0.7;
        // o[0][0][1] = 0.3;
        
        // o[0][1][0] = 0.3;
        // o[0][1][1] = 0.7;
        
        // o[1][0][0] = 0.3;
        // o[1][0][1] = 0.7;
        
        // o[1][1][0] = 0.7;
        // o[1][1][1] = 0.3;
        
        for (int i = 2; i < TNA; i++)
        {
            o[0][i][0] = ACC[i-2];
            o[0][i][1] = 1.0 - ACC[i-2];

            o[1][i][0] = 1.0 - ACC[i-2];
            o[1][i][1] = ACC[i-2];
        }

        // start
        double randunif = sample_unif();
        if (randunif <= 0.5)
        {
            curr_state = 0;
        }
        else 
        {
            curr_state = 1;
        }

        vector<int> epi_actions;
        vector<int> epi_obs;
        vector<double> epi_rewards;
        vector<int> epi_reward_obs;
        vector<int> epi_states;        

        actions.push_back(epi_actions);
        obs.push_back(epi_obs);
        rewards.push_back(epi_rewards);
        reward_obs.push_back(epi_reward_obs);
        states.push_back(epi_states);
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
        actions[epi].push_back(action);
        obs[epi].push_back(new_obs);
        reward_obs[epi].push_back(new_reward_obs);
        rewards[epi].push_back(ro_map[new_reward_obs]);
        states[epi].push_back(curr_state);
        curr_state = next_state;

        // cout << "action " << action << " " << prev_state << " -> " << next_state << endl;
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
        epi += 1;
        vector<int> epi_actions;
        vector<int> epi_obs;
        vector<double> epi_rewards;
        vector<int> epi_reward_obs;
        vector<int> epi_states;        

        actions.push_back(epi_actions);
        obs.push_back(epi_obs);
        rewards.push_back(epi_rewards);
        reward_obs.push_back(epi_reward_obs);
        states.push_back(epi_states);

        double randunif = sample_unif();
        if (randunif <= 0.5)
        {
            curr_state = 0;
        }
        else 
        {
            curr_state = 1;
        }            
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


// two states, left and right
// two actions, left and right
// going left in left gets some reward
// going right in right gets some reward
// you start off in left
// it's hard to transition to the right, but there's a high reward in the right
// observations are known, and transitions need to be learned
struct POMDP_OneArm
{
    int numstates;
    int numactions;
    int numobs;

    int epi = 0;

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

    // combination of o and ro, used as the observation matrix learned from MoM.
    // double o[TNS][TNA][TNO*TNR];

    vector<vector <int>> actions;
    vector<vector <int>> obs;
    vector<vector <double>> rewards;
    
    // keep track of reward obs
    vector<vector <int>> reward_obs;
    
    // useful for debugging
    vector<vector <int>> states;

    int curr_state;

    POMDP_OneArm()
        : numstates(TNS), numactions(TNA), numobs(TNO),
          gamma(0.99), rmax(BIG_REWARD)
    {
        // actions are: go left or go right
        // states are: left, right
        // obs are: hear left, hear right
        // obs are fixed

        // the only rewards are going left in left and going right in right,
        // except for a small chance of getting other rewards...
        r[0][0] = SMALL_REWARD * 0.8 + BIG_REWARD * 0.1 + 0 * 0.1;
        r[0][1] = 0 * 0.8 + SMALL_REWARD * 0.1 + BIG_REWARD * 0.1;

        r[1][0] = 0 * 0.8 + SMALL_REWARD * 0.1 + BIG_REWARD * 0.1;
        r[1][1] = BIG_REWARD * 0.8 + SMALL_REWARD * 0.1 + 0 * 0.1;

        // deterministic versionP
        // r[0][0] = SMALL_REWARD;
        // r[0][1] = 0;

        // r[1][0] = 0;
        // r[1][1] = 0.85;

        ro_map[0] = 0;
        ro_map[1] = SMALL_REWARD;
        ro_map[2] = BIG_REWARD;
        
        // reflected in the reward obs
        ro[0][0][0] = 0.1; 
        ro[0][0][1] = 0.8; // SMALL_REWARD
        ro[0][0][2] = 0.1;
        
        ro[0][1][0] = 0.8; // no reward
        ro[0][1][1] = 0.1;
        ro[0][1][2] = 0.1;
        
        ro[1][0][0] = 0.8; // no reward
        ro[1][0][1] = 0.1;
        ro[1][0][2] = 0.1;
        
        ro[1][1][0] = 0.1; 
        ro[1][1][1] = 0.1;
        ro[1][1][2] = 0.8; // BIG_REWARD

        // deterministic version:
        // ro[0][0][0] = 0.0; 
        // ro[0][0][1] = 1.0; // SMALL_REWARD
        // ro[0][0][2] = 0.0;
        
        // ro[0][1][0] = 1.0; // no reward
        // ro[0][1][1] = 0.0;
        // ro[0][1][2] = 0.0;
        
        // ro[1][0][0] = 1.0; // no reward
        // ro[1][0][1] = 0.0;
        // ro[1][0][2] = 0.0;
        
        // ro[1][1][0] = 0.0; 
        // ro[1][1][1] = 0.0;
        // ro[1][1][2] = 1.0; // BIG_REWARD
        
        // transitions are mostly deterministic except for trying to get to right from left
        t[0][0][0] = 1.0; // stay in the same state
        t[0][0][1] = 0.0;
        
        t[0][1][0] = 1.0 - SUCCESS_PROB; // hard to go right
        t[0][1][1] = SUCCESS_PROB;
        
        t[1][0][0] = 1.0 - LEARNING_EASE; // hard to learn if not going back left easily
        t[1][0][1] = LEARNING_EASE;
        
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

        // o[0][0][0] = 0;
        // o[0][0][1] = OBS_SUCCESS * 0.9;
        // o[0][0][2] = OBS_SUCCESS * 0.1;
        // o[0][0][3] = 0;
        // o[0][0][4] = (1.0 - OBS_SUCCESS) * 0.9;
        // o[0][0][5] = (1.0 - OBS_SUCCESS) * 0.1;
        
        // o[0][1][0] = OBS_SUCCESS * 0.9;
        // o[0][1][1] = OBS_SUCCESS * 0.1;
        // o[0][1][2] = 0;
        // o[0][1][3] = (1.0 - OBS_SUCCESS) * 0.9;
        // o[0][1][4] = (1.0 - OBS_SUCCESS) * 0.1;
        // o[0][1][5] = 0;
        
        // o[1][0][0] = (1.0 - OBS_SUCCESS) * 0.9;
        // o[1][0][1] = (1.0 - OBS_SUCCESS) * 0.1;
        // o[1][0][2] = 0;
        // o[1][0][3] = OBS_SUCCESS * 0.9;
        // o[1][0][4] = OBS_SUCCESS * 0.1;
        // o[1][0][5] = 0;   

        // o[1][1][0] = 0;
        // o[1][1][1] = (1.0 - OBS_SUCCESS) * 0.1;
        // o[1][1][2] = (1.0 - OBS_SUCCESS) * 0.9;
        // o[1][1][3] = 0;
        // o[1][1][4] = OBS_SUCCESS * 0.1;
        // o[1][1][5] = OBS_SUCCESS * 0.9;

        // start
        double randunif = sample_unif();
        if (randunif <= 1)
        {
            curr_state = 0;
        }
        else 
        {
            curr_state = 1;
        }

        vector<int> epi_actions;
        vector<int> epi_obs;
        vector<double> epi_rewards;
        vector<int> epi_reward_obs;
        vector<int> epi_states;        

        actions.push_back(epi_actions);
        obs.push_back(epi_obs);
        rewards.push_back(epi_rewards);
        reward_obs.push_back(epi_reward_obs);
        states.push_back(epi_states);
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
        actions[epi].push_back(action);
        obs[epi].push_back(new_obs);
        reward_obs[epi].push_back(new_reward_obs);
        rewards[epi].push_back(ro_map[new_reward_obs]);
        states[epi].push_back(curr_state);
        curr_state = next_state;

        //cout << "action " << action << " " << prev_state << " -> " << next_state << endl;
    }

    void new_episode()
    {
        epi += 1;
        vector<int> epi_actions;
        vector<int> epi_obs;
        vector<double> epi_rewards;
        vector<int> epi_reward_obs;
        vector<int> epi_states;        

        actions.push_back(epi_actions);
        obs.push_back(epi_obs);
        rewards.push_back(epi_rewards);
        reward_obs.push_back(epi_reward_obs);
        states.push_back(epi_states);

        double randunif = sample_unif();
        if (randunif <= 1.0)
        {
            curr_state = 0;
        }
        else 
        {
            curr_state = 1;
        }            
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
                double p = sample_unif() + 0.0000000001;
                //double p = sample_gamma() + 0.0000000001;
                //double p = 1.0;
                est_o[i][j][k] = p;
                total += p;
            }
            for (int k = 0; k < TNO; ++k)
            {
                est_o[i][j][k] /= total;
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
    if (0 and is_same<POMDP,POMDP_Tiger>::value)
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
double em(POMDP &pomdp, double (&est_t)[TNS][TNA][TNS], double (&err_t)[TNS][TNA][TNS], double (&est_o)[TNS][TNA][TNO], double (&err_o)[TNS][TNA][TNO], double (&est_ro)[TNS][TNA][TNR], double (&err_ro)[TNS][TNA][TNR], double (&est_r)[TNS][TNA], double (&opt_r)[TNS][TNA], double scale_t = 1.0, double scale_o = 1.0, double scale_ro = 1.0, double scale_r = 1.0)
{
    
    double pi [2] = {0.5, 0.5};
    
    const int num_iters = 400;
    for (int iters = 0; iters < num_iters; ++iters)
    {

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

        for (int epi = 0; epi < pomdp.actions.size(); ++epi)
        {   
            // the number of time steps that have occurred
            // the indices go from 0 to T-1

            // at time t, the current state is pomdp.states[epi][t]
            // pomdp.rewards[t] is the reward for the current state
            // pomdp.reward_obs[epi][t] is the reward obs for the current state
            // pomdp.obs[epi][t-1] is the current obs
            // pomdp.actions[epi][t] is the current action
            int T = pomdp.states[epi].size();
            if (T <= 0)
            {
                return 0;
            }

            double alpha[T][TNS];
            double beta[T][TNS];
            double gamma[T][TNS];

            // initialize the base cases of alpha and beta
            for (int i = 0; i < TNS; ++i)
            {
                if (EM_USE_RO)
                {
                    alpha[0][i] = est_ro[i][pomdp.actions[epi][0]][pomdp.reward_obs[epi][0]] * pi[i];
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
                        asum += alpha[ai-1][j]*est_t[j][pomdp.actions[epi][ai-1]][i];
                        if (EM_USE_RO)
                        {
                            beta[bi][i] += beta[bi+1][j]*est_t[i][pomdp.actions[epi][bi]][j]*est_o[j][pomdp.actions[epi][bi]][pomdp.obs[epi][bi]]*est_ro[j][pomdp.actions[epi][bi+1]][pomdp.reward_obs[epi][bi+1]];
                        }
                        else
                        {
                            beta[bi][i] += beta[bi+1][j]*est_t[i][pomdp.actions[epi][bi]][j]*est_o[j][pomdp.actions[epi][bi]][pomdp.obs[epi][bi]];
                        }
                    }
                    if (EM_USE_RO)
                    {
                        alpha[ai][i] = asum * est_o[i][pomdp.actions[epi][ai-1]][pomdp.obs[epi][ai-1]] * est_ro[i][pomdp.actions[epi][ai]][pomdp.reward_obs[epi][ai]];
                    }
                    else
                    {
                      alpha[ai][i] = asum * est_o[i][pomdp.actions[epi][ai-1]][pomdp.obs[epi][ai-1]];
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
                    gamma[t][i] /= sum;;
                }
            }
            
            // calculate gamma_action_sum variables
            for (int t = 0; t < T; ++t)
            {
                // current action
                int cact = pomdp.actions[epi][t];
                
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
                    int cact = pomdp.actions[epi][t];
                    
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
                                top = alpha[t][x]*est_t[x][cact][z]*est_o[z][cact][pomdp.obs[epi][t]]*beta[t+1][z]*est_ro[z][pomdp.actions[epi][t+1]][pomdp.reward_obs[epi][t+1]];
                            }
                            else
                            {
                                top = alpha[t][x]*est_t[x][cact][z]*est_o[z][cact][pomdp.obs[epi][t]]*beta[t+1][z];
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
            }
        
            // estimate obs
            if (EST_O)
            {
                // add up all things for the obs
                for (int t = 1; t < T; ++t)
                {
                    // previous action
                    int pact = pomdp.actions[epi][t-1];
                    
                    // current obs
                    int cobs = pomdp.obs[epi][t-1];
                    
                    // calculate gamma_action_sum
                    for (int x = 0; x < TNS; ++x)
                    {
                        // update only the entries corresponding to the action at this timestep
                        obs_sum[x][pact][cobs] += gamma[t][x];
                    }
                }
            }
            
            // estimate reward obs
            if (EST_RO)
            {
                // add up all things for the reward obs
                for (int t = 0; t < T; ++t)
                {
                    // current action
                    int cact = pomdp.actions[epi][t];
                    
                    // current reward obs
                    int crobs = pomdp.reward_obs[epi][t];
                    
                    // calculate gamma_action_sum
                    for (int x = 0; x < TNS; ++x)
                    {
                        // update only the entries corresponding to the action at this timestep
                        rho_sum[x][cact][crobs] += gamma[t][x];
                    }
                }
            }
            
            // estimate the rewards
            if (EST_R)
            {
                // add up all things for the estimated rewards
                for (int t = 0; t < T; ++t)
                {
                    // current action
                    int cact = pomdp.actions[epi][t];
                    
                    // calculate ex_reward
                    for (int x = 0; x < TNS; ++x)
                    {
                        // weighted by the belief prob
                        ex_reward[x][cact] += pomdp.rewards[epi][t] * gamma[t][x];
                    }
                }
            }
        }

        // M STEP: Use results averaged over all episodes to estimate the parameters.
        
        // some parameter for the confidence intervals
        double const confidence_alpha = 0.9;
 
        if (EST_T)
        {
            // now it's easy to compute the estimated probs using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 0; y < TNA; ++y)
                {
                    // compute the confidence intervals

                    double fake_count = 0;
                    if (scale_t == 1.0)
                    {
                        fake_count = gamma_action_sum[x][y];
                    }
                    else 
                    {
                        for (int ep = 0; ep < pomdp.actions.size(); ep++)
                        {
                            for (int c = 0; c < pomdp.actions[ep].size(); c++)
                            {
                                if (y == pomdp.actions[ep][c]) {fake_count += 1;}
                            }
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
        if (EST_O)
        {
            // now it's easy to compute the estimated probs using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 0; y < TNA; ++y)
                {
                    // compute the confidence intervals

                    double fake_count = 0;
                    if (scale_o == 1.0)
                    {
                        fake_count = gamma_action_sum[x][y];
                    }
                    else 
                    {
                        for (int ep = 0; ep < pomdp.actions.size(); ep++)
                        {
                            for (int c = 0; c < pomdp.actions[ep].size(); c++)
                            {
                                if (y == pomdp.actions[ep][c]) {fake_count += 1;}
                            }
                        }
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

        if (EST_RO)
        {
            // now it's easy to compute the estimated probs using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 0; y < TNA; ++y)
                {
                    // compute the confidence intervals
                    
                    double fake_count = 0;
                    if (scale_ro == 1.0)
                    {
                        fake_count = gamma_action_sum[x][y];
                    }
                    else 
                    {
                        for (int ep = 0; ep < pomdp.actions.size(); ep++)
                        {
                            for (int c = 0; c < pomdp.actions[ep].size(); c++)
                            {
                                if (y == pomdp.actions[ep][c]) {fake_count += 1;}
                            }
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

        if (EST_R)
        {
            // now it's easy to compute the estimated rewards using the sum variables above
            // can also get the fake confidence intervals from the expected counts
            for (int x = 0; x < TNS; ++x)
            {
                for (int y = 0; y < TNA; ++y)
                {
                    double fake_count = 0;
                    if (scale_r == 1.0)
                    {
                        fake_count = gamma_action_sum[x][y];
                    }
                    else 
                    {
                        for (int ep = 0; ep < pomdp.actions.size(); ep++)
                        {
                            for (int c = 0; c < pomdp.actions[ep].size(); c++)
                            {
                                if (y == pomdp.actions[ep][c]) {fake_count += 1;}
                            }
                        }
                    }
                    double ci_radius = fake_count >= 1.0 ? REWARD_GAP*sqrt((0.5/(scale_r * fake_count))*log (2.0/confidence_alpha)) : REWARD_GAP;
                    
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
    // if (0)
    // {
    //     cout << "em called" << endl;
    //     for (int l = 0; l <= T; l++){
    //         cout << "sim step " << l << ", s " << pomdp.states[epi][l] << ", r " << pomdp.rewards[l] << ", a " << pomdp.actions[epi][l] << ", o " << pomdp.obs[epi][l] << ": ";
    //         for (int i = 0; i < pomdp.numstates; i++) {
    //             cout << (gamma[l][i]) << " ";
    //         }
    //         cout << "| ";
    //         for (int i = 0; i < pomdp.numstates; i++) {
    //             cout << (alpha[l][i]) << " ";
    //         }
    //         cout << "| ";
    //         for (int i = 0; i < pomdp.numstates; i++) {
    //             cout << (beta[l][i]) << " ";
    //         }
    //         cout << endl;
    //     }
    // }
    
    double log_const = log(1.0);
    double ll = log(0.0);

    // calculate the log likelihood by recomputing the alphas and keeping the constants around

    for (int epi = 0; epi < pomdp.states.size(); ++epi)
    {
        int T = pomdp.states[epi].size();
        double alpha[T][TNS];

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
                    asum += alpha[ai-1][j]*est_t[j][pomdp.actions[epi][ai-1]][i];
                }
                alpha[ai][i] = asum * est_o[i][pomdp.actions[epi][ai-1]][pomdp.obs[epi][ai-1]];
                
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
        for (int i = 0; i < TNS; ++i)
        {
            ll = mylogadd(ll, log(alpha[T-1][i]));
        }
        ll = mylogmul(ll, log_const);
    }
    
    return ll;
}

double likelihood(POMDP &pomdp, double est_o[TNS][TNA][TNO], double est_t[TNS][TNA][TNS])   
{
    double log_const = log(1.0);
    double ll = log(0.0);
    for (int epi = 0; epi < pomdp.states.size(); ++epi)
    {

        const int T = pomdp.states[epi].size();
        double pi[2] = {0.5, 0.5};
        if (T <= 0)
        {
            return 0;
        }
        
        double alpha[T][TNS];
        
        // calculate the log likelihood by recomputing the alphas and keeping the constants around
        // initialize the base cases of alpha
        for (int i = 0; i < TNS; ++i)
        {
            alpha[0][i] = pi[i];
        }
        
        // recursively build up alpha and beta from previous values
        for (int t = 1; t < T; ++t)
        {
            // alpha goes forward
            int ai = t;
            // to normalize alpha
            double a_denom = 0;
            // do the recursive update
            for (int i = 0; i < TNS; ++i) 
            {
                double asum = 0;
                for (int j = 0; j < TNS; ++j)
                {
                    asum += alpha[ai-1][j]*est_t[j][pomdp.actions[epi][ai-1]][i];
                }
                alpha[ai][i] = asum * est_o[i][pomdp.actions[epi][ai-1]][pomdp.obs[epi][ai-1]];
                
                a_denom += alpha[ai][i];
            }
            if (not (a_denom > 0 or a_denom <= 0))
            {
                for (int x = 0; x < TNS; x++)
                {
                    for (int y = 0; y < TNA; y++)
                    {
                        for (int z = 0; z < TNO; z++)
                        {
                            cout << x << y << z << " : " << est_t[x][y][z] << endl;
                        }
                    }
                }
                for (int x = 0; x < TNS; x++)
                {
                    for (int y = 0; y < TNA; y++)
                    {
                        for (int z = 0; z < TNO; z++)
                        {
                            cout << x << y << z << " : " << est_o[x][y][z] << endl;
                        }
                    }
                }
            }
            if (a_denom <= 0)
            {
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
        for (int i = 0; i < TNS; ++i)
        {
            ll = mylogadd(ll, log(alpha[T-1][i]));
        }
        ll = mylogmul(ll, log_const);
    }
    return ll;
}

void test_random()
{
    // test random number generators
    cout << "Using rand() is in [" << 0 << "," << RAND_MAX << "]" << endl;
    cout << "Using mt19937() is in [" << mt19937::min() << "," << mt19937::max() << "]" << endl;
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
    POMDP pomdp;
    Planning<POMDP,double[TNS][TNA][TNS],double[TNS][TNA][TNO],double[TNS][TNA]> plan(pomdp);
    
    double est_t[TNS][TNA][TNS];
    double err_t[TNS][TNA][TNS];
    
    double est_o[TNS][TNA][TNO];
    double err_o[TNS][TNA][TNO];
    
    //double est_ro[TNS][TNA][TNR];
    //double err_ro[TNS][TNA][TNR];
    
    //double est_r[TNS][TNA];
    double opt_r[TNS][TNA];
    
    // initial initialization
    zero_out(err_t);
    zero_out(err_o);
    // set to be the same as true params
    copy_matrix(pomdp.t, est_t);
    copy_matrix(pomdp.o, est_o);
    copy_matrix(pomdp.r, opt_r);
    
    //sample_seed(time(0));
    //initialize(pomdp, est_t, est_o, est_ro);
    
    print_matrix(est_t);
    
    // let's make a purposefully bad transition matrix but with
    //tr[0][0][0] = 0.99; // stay in the same state
    //tr[0][0][1] = 0.01;
    
    //err[0][0][0] = 0.05; // a small chance to include more enticing probs
    //err[0][0][1] = 0.05;
    
    //tr[0][1][0] = 0.65; // hard to go right
    //tr[0][1][1] = 0.35;
    
    //err[0][1][0] = 0.00; // a small chance to include correct probs
    //err[0][1][1] = 0.00;
    
    //tr[1][0][0] = 1.0; // easily go back to left
    //tr[1][0][1] = 0.0;
    
    //tr[1][1][0] = 0.0; // stay right
    //tr[1][1][1] = 1.0;
    
    //err[1][1][0] = 0.5; // wrong estimate possible
    //err[1][1][1] = 0.5;
    
    //est_r[0][0] = 0.6;
    //est_r[0][1] = 0.0;
    //est_r[1][0] = 1.0;
    //est_r[1][1] = 1.0;
    
    int next_action = plan.backup_plan(est_t, err_t, est_o, err_o, opt_r, true, 40);
    assert (next_action >= 0);
    plan.print_points();
    print_matrix(plan.opt_t);
    print_matrix(plan.opt_z);
}

// testing the correctness of em
// seems to work now
// void test_em(string const &l2_file, string const &l2_err_file, string const &linf_file, string const &linf_err_file)
// {
//     unsigned seed = 0;
//     //unsigned seed = time(0);
//     cout << "seed " << seed << endl;
//     sample_seed(seed);

//     int reps = 1;
    
//     int steps_start = 500;
//     int numsteps = 1;
    
//     vector<double> l2_dists(numsteps);
//     vector<double> linf_dists(numsteps);
    
//     vector<double> l2_dists_err(numsteps);
//     vector<double> linf_dists_err(numsteps);

//     double tr[TNS][TNA][TNS];
//     double err[TNS][TNA][TNS];
    
//     double est_o[TNS][TNA][TNS];
//     double err_o[TNS][TNA][TNS];
    
//     double est_ro[TNS][TNA][TNR];
//     double err_ro[TNS][TNA][TNR];
    
//     double est_r[TNS][TNA];
//     double opt_r[TNS][TNA];
    
//     for (int x = 0; x < TNS; x++) {
//         for (int y = 0; y < TNA; y++) {
//             est_r[x][y] = 0.0;
//             opt_r[x][y] = 0.0;
//             for (in(pt z = 0; z < TNS; z++) {
//                 tr[x][y][z] = 0;
//                 err[x][y][z] = 0;
//             }
//             for (int z = 0; z < TNO; z++) {
//                 est_o[x][y][z] = 0;
//                 err_o[x][y][z] = 0;
//             }
//             for (int z = 0; z < TNR; z++) {
//                 est_ro[x][y][z] = 0;
//                 err_ro[x][y][z] = 0;
//             }
//         }
//     }
    
//     for (int rep = 0; rep < reps; ++rep)
//     {
//         clock_t start = clock();
//         POMDP pomdp;
        
//         double ll = 0.0;
//         // advance the pomdp
//         for (int i = 0; i < steps_start; ++i)
//         {
//             int next_action = sample_int(0, TNA-1);
//             //int next_action = sample_unif() > 0.5;
//             //int next_action = (i / 10) % 2;
//             assert (next_action >= 0);
//             pomdp.step(next_action);
//         }
//         //sample_seed(time(0));
//         for (int i = 0; i < numsteps; ++i)
//         {
//             // do em for the next iteration after planning
//             initialize(pomdp,tr,est_o,est_ro);
//             ll = em(pomdp, tr, err, est_o, err_o, est_ro, err_ro, est_r, opt_r);
            
//             int next_action = sample_int(0, TNA-1);
//             //int next_action = sample_unif() > 0.5;
//             //int next_action = (i / 10) % 2;
//             assert (next_action >= 0);
//             pomdp.step(next_action);
            
//             // keep track of distances to the real parameters
//             // and how big the confidence intervals are
            
//             double l2dist = 0.0;
//             l2dist += l2_dist_squared(tr, pomdp.t);
//             l2dist += l2_dist_squared(est_o, pomdp.o);
//             l2dist += l2_dist_squared(est_ro, pomdp.ro);
//             //l2dist += l2_dist_squared(est_r, pomdp.r);
//             l2dist = sqrt(l2dist);
//             l2_dists.at(i) += l2dist;
            
//             double linfdist = 0.0;
//             linfdist = fmax(linfdist, linf_dist(tr, pomdp.t));
//             linfdist = fmax(linfdist, linf_dist(est_o, pomdp.o));
//             linfdist = fmax(linfdist, linf_dist(est_ro, pomdp.ro));
//             //linfdist = fmax(linfdist, linf_dist(est_r, pomdp.r));
//             linf_dists.at(i) += linfdist;
            
//             l2dist = sqrt(l2_dist_squared(est_r, opt_r));
//             l2_dists_err.at(i) += l2dist;
            
//             linfdist = linf_dist(est_r, opt_r);
//             linf_dists_err.at(i) += linfdist;
            
//         }
//         clock_t elapsed = clock() - start;
//         cout << "Elapsed " << 1.0*elapsed / CLOCKS_PER_SEC << endl;
//         if (1)
//         {
//             // show some stats on the run
//             cout << "states" << endl;
//             for (size_t i = 0; i < pomdp.states[epi].size(); ++i)
//             {
//                 cout << pomdp.states[epi][i] << " ";
//             }
//             cout << endl;
//             cout << "actions" << endl;
//             for (size_t i = 0; i < pomdp.actions[epi].size(); ++i)
//             {
//                 cout << pomdp.actions[epi][i] << " ";
//             }
//             cout << endl;
//             cout << "obs" << endl;
//             for (size_t i = 0; i < pomdp.obs[epi].size(); ++i)
//             {
//                 cout << pomdp.obs[epi][i] << " ";
//             }
//             cout << endl;
//             cout << "reward obs" << endl;
//             for (size_t i = 0; i < pomdp.reward_obs[epi].size(); ++i)
//             {
//                 cout << pomdp.reward_obs[epi][i] << " ";
//             }
//             cout << endl;
                
//             // show the very last em step
//             cout << "ll = " << ll << endl;
            
//             cout << "esimated transitions" << endl;
//             print_both(tr, err);
//             cout << "estimated obs" << endl;
//             print_both(est_o, err_o);
//             cout << "esimated reward obs" << endl;
//             print_both(est_ro, err_ro);
//             cout << "estimated rewards" << endl;
//             print_both(est_r, opt_r);
//             //cout << "seed " << seed << endl;
//         }
//     }
//     ofstream l2_out(l2_file);
//     ofstream linf_out(linf_file);
//     ofstream l2_err_out(l2_err_file);
//     ofstream linf_err_out(linf_err_file);
//     for (size_t i = 0; i < l2_dists.size(); ++i)
//     {
//         l2_dists.at(i) /= reps;
//         linf_dists.at(i) /= reps;
        
//         l2_dists_err.at(i) /= reps;
//         linf_dists_err.at(i) /= reps;
        
//         l2_out << l2_dists.at(i) << endl;
//         linf_out << linf_dists.at(i) << endl;
        
//         l2_err_out << l2_dists_err.at(i) << endl;
//         linf_err_out << linf_dists_err.at(i) << endl;
//     }
//     l2_out.close();
//     linf_out.close();
//     l2_err_out.close();
//     linf_err_out.close();
    
//     cout << "l2 dists" << endl;
//     print_vector(l2_dists);
//     cout << "l2 dists err" << endl;
//     print_vector(l2_dists_err);
//     cout << "linf dists" << endl;
//     print_vector(linf_dists);
//     cout << "linf dists err" << endl;
//     print_vector(linf_dists_err);
// }

void test_opt(bool use_opt, bool use_mom, string const &reward_out, string const &cumreward_out, string const &actions_out, double decay = -1)
{
    unsigned initial_seed = 745898798;
    //unsigned initial_seed = time(0);
    
    // generate a bunch of seeds, one for each rep
    sample_seed(initial_seed);

    vector<unsigned> seeds;
    // seeds.push_back(3303811027);
    // seeds.push_back(1610352249);
    seeds.push_back(1610352249);
    for (int i = 0; i < 20; ++i)
    {
        seeds.push_back(sample_rand());
    }
    //seeds.push_back(2546248239);
    
    int reps = seeds.size();
    int eps = 400;
    int max_steps_per_ep = 50;
    int steps = max_steps_per_ep * eps;
    double sum_rewards = 0;
    double prev_sum = 0;
    int nrandtriples = 0;

    // number of EM random restarts
    int restarts = 5;

    vector<double> rs(steps, 0.0);
    vector<double> eprs(eps, 0.0);

    ofstream histogram_r(cumreward_out + ".txt");
    ofstream output_a(actions_out + ".txt");

    mat I = eye<mat>(TNO*TNR, TNO*TNR);
    
    double est_t[TNS][TNA][TNS];
    double err_t[TNS][TNA][TNS];
    
    double est_o[TNS][TNA][TNO];
    double err_o[TNS][TNA][TNO];
    
    double est_ro[TNS][TNA][TNR];
    double err_ro[TNS][TNA][TNR];
    
    double est_r[TNS][TNA];
    double opt_r[TNS][TNA];

    double ro_map[TNR];

    double scale_t, scale_o, scale_ro, scale_r;
    if (use_mom)
    {
        scale_t = scale_o = scale_ro = scale_r = 0.5;
    }
    else
    {
        scale_t = scale_o = scale_ro = scale_r = 1.0;
    }

    for (int rep = 0; rep < reps; ++rep)
    {
        // scale -= 0.01;
        // keep track of episodes
        int curr_ep = 0;
        int curr_ep_step = -1;
        int curr_step = -1;
        int last_triple = -1;
        int momcount[TNA] = {0, 0};

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
        bool mom_updated[2] = {false, false};
    
        unsigned seed = seeds.at(rep);
        sample_seed(seed);
        cout << "---- Start rep " << rep << endl;
        cout << "seed " << seed << endl;
        rept = clock();
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
                    err_o[x][y][z] = 1;
                }
                for (int z = 0; z < TNR; z++) {
                    est_ro[x][y][z] = 0;
                    err_ro[x][y][z] = 1;
                }
            }
        }
        for (int x = 0; x < TNR; x++)
        {
            ro_map[x] = pomdp.rmax;
        }
        for (int i = 0; i < nrandtriples; i++)
        {
            int a;
            // double randunif = sample_unif();
            // if (randunif <= 0.5)
            // {
            //     a = 1;
            // }
            // else 
            // {
            //     a = 0;
            // }
            a = sample_int(0, 1);
            for (int j = 0; j < 4; j++)
            {
                pomdp.step(a);
            }
            // a = pomdp.actions[curr_ep][3*i + 1];
            newobs = randu<cube>(TNO*TNR, obs.at(a).n_cols + 1, 3);
            if (newobs.n_cols > 1)
            { 
                newobs.subcube(0, 0, 0, TNO*TNR - 1, obs.at(a).n_cols - 1, 2) = obs.at(a);
            }
            for (int k = 0; k < 3; k++)
            {
                newobs.slice(k).col(obs.at(a).n_cols) = I.col(pomdp.obs[curr_ep][3*i + k]*TNR + pomdp.reward_obs[curr_ep][3*i + k]);
            } 
            obs.at(a) = newobs;
            // curr_ep += 1;
            // pomdp.new_episode();
        }
        // curr_ep += 1;
        // pomdp.new_episode();
        initialize(pomdp, est_t, est_o, est_ro);
        Planning<POMDP,double[TNS][TNA][TNS],double[TNS][TNA][TNO],double[TNS][TNA]> plan(pomdp);
        for (int iter = 0; iter < steps; iter++) {
            ++curr_ep_step;
            ++curr_step;
             //cout << "---------- Iteration " << iter+1 << " ----------" << endl;
            //cout << "Curr Belief -- ";
            //print_vector(plan.curr_belief);
            // t = clock();
            // if nonoptimistic
            if (not use_opt) {
                // zero out the confidence intervals
                zero_out(err_t);
                zero_out(err_o);
                // set opt r to be same as est r
                copy_matrix(est_r, opt_r);
            }
            int next_action = -1;

            // ######################### 
            // Use this when we want to model the optimal policy.
            // if (iter == 0)
            // {
            //     next_action = plan.backup_plan(pomdp.t, err_t, pomdp.o, err_o, pomdp.r, true, 40);
            // }
            // else
            // {
            //     next_action = plan.backup_plan(pomdp.t, err_t, pomdp.o, err_o, pomdp.r, false, 1);
            // }
            // ##########################

            if (model_updated)
            {
                next_action = plan.backup_plan(est_t, err_t, est_o, err_o, opt_r, true, 40);
                model_updated = false;
                // cout << "estimated obs" << endl;
                // print_both(est_o, err_o);
                mom_updated[0] = false;
                mom_updated[1] = false;
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
            // egreedy random action with decay
            // but stop egreedy for the last 2000 steps
            if (decay > 0.0 and iter + 2000 < steps)
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
            ro_map[pomdp.reward_obs[curr_ep].back()] = pomdp.rewards[curr_ep].back();
            //cout << "Curr Belief -- ";
            //print_vector(plan.curr_belief);

            // if (pomdp.actions[epi][iter] == pomdp.actions[epi][iter - 1] and pomdp.actions[epi][iter - 1] == pomdp.actions[epi][iter - 2] and last_triple < iter - 2 and curr_ep_step >= 2) 
            // {
            //     int a = pomdp.actions[epi][iter];
            //     newobs = randu<cube>(TNO, obs.at(a).n_cols + 1, 3);
            //     if (newobs.n_cols > 1)
            //     { 
            //         newobs.subcube(0, 0, 0, TNO - 1, obs.at(a).n_cols - 1, 2) = obs.at(a);
            //     }
            //     for (int i = 0; i < 3; i++)
            //     {
            //         newobs.slice(i).col(obs.at(a).n_cols) = I.col(pomdp.obs[epi][iter-2+i]);
            //     }
            //     obs.at(a) = newobs;
            //     last_triple = iter;
            // }

            // Version when not using reward_obs.
            // if (curr_ep_step >= 3 and pomdp.actions[curr_ep][curr_ep_step] == pomdp.actions[curr_ep][curr_ep_step - 1] and last_triple < iter - 2) 

            // Modified for using reward_obs.
            // if (curr_ep_step >= 3 and pomdp.actions[curr_ep][curr_ep_step] == pomdp.actions[curr_ep][curr_ep_step - 1] and pomdp.actions[curr_ep][curr_ep_step - 1] == pomdp.actions[curr_ep][curr_ep_step - 2] and pomdp.actions[curr_ep][curr_ep_step - 2] == pomdp.actions[curr_ep][curr_ep_step - 3]  and last_triple < iter - 2) 
            // {
            //     int a = pomdp.actions[curr_ep][curr_ep_step];
            //     newobs = randu<cube>(TNO*TNR, obs.at(a).n_cols + 1, 3);
            //     if (newobs.n_cols > 1)
            //     { 
            //         newobs.subcube(0, 0, 0, TNO*TNR - 1, obs.at(a).n_cols - 1, 2) = obs.at(a);
            //     }
            //     for (int i = 0; i < 3; i++)
            //     {
            //         newobs.slice(i).col(obs.at(a).n_cols) = I.col(pomdp.obs[curr_ep][curr_ep_step-3+i]*TNR + pomdp.reward_obs[curr_ep][curr_ep_step-2+i]);
            //     }
            //     obs.at(a) = newobs;
            //     last_triple = iter;
            //     update_model = true;
            // }
            // else
            // {
            //     update_model = false;
            // }

            // for (int i = 0; i < TNA; i++)
            // {
            //     cout << "Samples for action " << i << ": " << obs.at(i).n_cols << endl;
            // }

            // update beliefs for next step
            plan.belief_update_full();

            // int initial_burn_in = 0;
            // if (1)
            // if (update_model)
            if (iter % 50 == 0)
            // if ((curr_ep == initial_burn_in or (curr_ep > initial_burn_in and (curr_ep-initial_burn_in) % 50  == 0)) and curr_ep_step == 0)            
            {
                // initialize(pomdp, est_t, est_o, est_ro);
                // double ll = em(pomdp, est_t, err_t, est_o, err_o, est_ro, err_ro, est_r, opt_r, scale_t, scale_o, scale_ro, scale_r);

                double tmp_est_t[TNS][TNA][TNS];
                double tmp_err_t[TNS][TNA][TNS];
                
                double tmp_est_o[TNS][TNA][TNO];
                double tmp_err_o[TNS][TNA][TNO];
                
                double tmp_est_ro[TNS][TNA][TNR];
                double tmp_err_ro[TNS][TNA][TNR];
                
                double tmp_est_r[TNS][TNA];
                double tmp_opt_r[TNS][TNA];
                double best_ll = - numeric_limits<double>::infinity();
                double ll = 0;
                for (int restart = 0; restart < restarts; restart++)
                {
                    initialize(pomdp, tmp_est_t, tmp_est_o, tmp_est_ro);
                    ll = em(pomdp, tmp_est_t, tmp_err_t, tmp_est_o, tmp_err_o, tmp_est_ro, tmp_err_ro, tmp_est_r, tmp_opt_r, scale_t, scale_o, scale_ro, scale_r);
                    if (ll > best_ll)
                    {
                        // cout << best_ll << " " << ll << endl;
                        best_ll = ll;
                        copy_matrix(tmp_est_t, est_t);
                        copy_matrix(tmp_err_t, err_t);
                        copy_matrix(tmp_est_o, est_o);
                        copy_matrix(tmp_err_o, err_o);
                        copy_matrix(tmp_est_ro, est_ro);
                        copy_matrix(tmp_err_ro, err_ro);
                        copy_matrix(tmp_est_r, est_r);
                        copy_matrix(tmp_opt_r, opt_r);
                    }
                }
                // cout << "Log likelihood: " << best_ll << endl;
                // cout << "True Log likelihood: " << likelihood(pomdp, pomdp.o, pomdp.t) << endl;
                // if (est_o[0][0][0] <= 0.4 and est_o[0][1][0] <= 0.4)
                // {
                //     double temp_o[TNS][TNA][TNO];
                //     double temp_t[TNS][TNA][TNS];
                //     // print_matrix(est_o);
                //     for (int i = 0; i < TNS; i++)
                //     {
                //         for (int j = 0; j < TNA; j++)
                //         {                    
                //             for (int k = 0; k < TNO; k++)
                //             {
                //                 temp_o[i][j][k] = est_o[(i + 1) % 2][j][k];
                //             }
                //             for (int k = 0; k < TNS; k++)
                //             {
                //                 temp_t[i][j][k] = est_t[(i + 1) % 2][j][(k + 1) % 2];
                //             }
                //         }
                //     }
                //     for (int i = 0; i < TNS; i++)
                //     {
                //         for (int j = 0; j < TNA; j++)
                //         { 
                //             for (int k = 0; k < TNO; k++)
                //             {
                //                 est_o[i][j][k] = temp_o[i][j][k];
                //             }
                //             for (int k = 0; k < TNS; k++)
                //             {
                //                 est_t[i][j][k] = temp_t[i][j][k];
                //             }
                //         }
                //     }
                // }
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
            // if ((iter % 50 == 0))
            // {
            //     cout << "EM Curr episode: " << curr_ep << endl;
            //     print_both(est_o, err_o);
            // }
            if (use_mom and update_model)
            // if (use_mom and iter % 50 == 0)
            // if ((curr_ep == initial_burn_in or (curr_ep > initial_burn_in and (curr_ep-initial_burn_in) % 50  == 0)) and curr_ep_step == 0)
            {
                int ENO = TNO * TNR;
                double new_o[TNS][TNA][TNO];
                double new_ro[TNS][TNA][TNR];
                double new_t[TNS][TNA][TNS];
                double best_o[TNS][TNA][TNO];
                double best_ro[TNS][TNA][TNR];
                double best_t[TNS][TNA][TNS];

                for (int i = 0; i < TNA; i++)
                {
                    mat o_mat = zeros<mat>(TNS, ENO);
                    mat t_mat = zeros<mat>(TNS, TNS);
                    if (obs.at(i).n_cols > 0)
                    {
                        LearnHMM(o_mat, t_mat, obs.at(i), TNS);
                        urowvec nonzeros = any(o_mat, 0);
                        if (all(nonzeros))
                        {
                            momcount[i] += 1;
                            for (int j = 0; j < TNS; j++)
                            {   
                                for (int k = 0; k < TNO; k++)
                                {
                                    est_o[j][i][k] = 0.0;
                                }     
                                for (int l = 0; l < TNR; l++)
                                {
                                    est_ro[j][i][l] = 0.0;
                                }          
                                for (int k = 0; k < TNO; k++)
                                {
                                    for (int l = 0; l < TNR; l++)
                                    {
                                        est_o[j][i][k] += o_mat(k*TNR + l, j);
                                        est_ro[j][i][l] += o_mat(k*TNR + l, j);
                                    }
                                }
                                for (int k = 0; k < TNS; k++)
                                {
                                    est_t[j][i][k] = t_mat(k, j);
                                }
                            }
                            mom_updated[i] = true;
                        }
                    }
                }
                // if (est_o[0][0][0] <= 0.4)
                // {
                //     double temp_o[TNS][TNA][TNO];
                //     double temp_t[TNS][TNA][TNO];
                //     // print_matrix(est_o);
                //     for (int i = 0; i < TNS; i++)
                //     {                  
                //         for (int k = 0; k < TNO; k++)
                //         {
                //             temp_o[i][0][k] = est_o[(i + 1) % 2][0][k];
                //         }
                //         for (int k = 0; k < TNS; k++)
                //         {
                //             temp_t[i][0][k] = est_t[(i + 1) % 2][0][(k + 1) % 2];
                //         }
                //     }
                //     for (int i = 0; i < TNS; i++)
                //     {
                //         for (int k = 0; k < TNO; k++)
                //         {
                //             est_o[i][0][k] = temp_o[i][0][k];
                //         }
                //         for (int k = 0; k < TNS; k++)
                //         {
                //             est_t[i][0][k] = temp_t[i][0][k];
                //         }
                //     }
                // }
                // if (est_o[0][1][0] <= 0.4)
                // {
                //     double temp_o[TNS][TNA][TNO];
                //     double temp_t[TNS][TNA][TNO];
                //     // print_matrix(est_o);
                //     for (int i = 0; i < TNS; i++)
                //     {                    
                //         for (int k = 0; k < TNO; k++)
                //         {
                //             temp_o[i][1][k] = est_o[(i + 1) % 2][1][k];
                //         }
                //         for (int k = 0; k < TNS; k++)
                //         {
                //             temp_t[i][1][k] = est_t[(i + 1) % 2][1][(k + 1) % 2];
                //         }
                //     }
                //     for (int i = 0; i < TNS; i++)
                //     {
                //         for (int k = 0; k < TNO; k++)
                //         {
                //             est_o[i][1][k] = temp_o[i][1][k];
                //         }
                //         for (int k = 0; k < TNS; k++)
                //         {
                //             est_t[i][1][k] = temp_t[i][1][k];
                //         }
                //     }
                // }
                // cout << "MoM ll: " << likelihood(pomdp, est_o, est_t) << endl;
                int statearray[TNA][TNS];

                for (int i = 0; i < TNS; i++)
                {
                    statearray[0][i] = i;
                    statearray[1][i] = i;
                }
                copy_matrix(est_o, best_o);
                copy_matrix(est_t, best_t);
                copy_matrix(est_ro, best_ro);
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
                                    if (j == 0)
                                    {
                                        new_o[i][j][k] = est_o[statearray[0][i]][j][k]; 
                                    }
                                    else
                                    {
                                        new_o[i][j][k] = est_o[statearray[1][i]][j][k]; 
                                    }
                                }
                                for (int k = 0; k < TNR; k++)
                                {
                                    if (j == 0)
                                    {
                                        new_ro[i][j][k] = est_ro[statearray[0][i]][j][k]; 
                                    }
                                    else
                                    {
                                        new_ro[i][j][k] = est_ro[statearray[1][i]][j][k]; 
                                    }
                                }
                                for (int k = 0; k < TNS; k++)
                                {
                                    if (j == 0)
                                    {
                                        new_t[i][j][k] = est_t[statearray[0][i]][j][statearray[0][k]]; 
                                    }
                                    else
                                    {
                                        new_t[i][j][k] = est_t[statearray[1][i]][j][statearray[1][k]]; 
                                    }                        
                                }
                            }
                        }
                        // if (new_o[0][0][0] <= 0.5)
                        // {
                        //     double temp_o[TNS][TNA][TNO];
                        //     double temp_t[TNS][TNA][TNO];
                        //     // print_matrix(est_o);
                        //     for (int i = 0; i < TNS; i++)
                        //     {                  
                        //         for (int k = 0; k < TNO; k++)
                        //         {
                        //             temp_o[i][0][k] = new_o[(i + 1) % 2][0][k];
                        //         }
                        //         for (int k = 0; k < TNS; k++)
                        //         {
                        //             temp_t[i][0][k] = new_t[(i + 1) % 2][0][(k + 1) % 2];
                        //         }
                        //     }
                        //     for (int i = 0; i < TNS; i++)
                        //     {
                        //         for (int k = 0; k < TNO; k++)
                        //         {
                        //             new_o[i][0][k] = temp_o[i][0][k];
                        //         }
                        //         for (int k = 0; k < TNS; k++)
                        //         {
                        //             new_t[i][0][k] = temp_t[i][0][k];
                        //         }
                        //     }
                        // }
                        // if (new_o[0][1][0] <= 0.5)
                        // {
                        //     double temp_o[TNS][TNA][TNO];
                        //     double temp_t[TNS][TNA][TNO];
                        //     // print_matrix(est_o);
                        //     for (int i = 0; i < TNS; i++)
                        //     {                    
                        //         for (int k = 0; k < TNO; k++)
                        //         {
                        //             temp_o[i][1][k] = new_o[(i + 1) % 2][1][k];
                        //         }
                        //         for (int k = 0; k < TNS; k++)
                        //         {
                        //             temp_t[i][1][k] = new_t[(i + 1) % 2][1][(k + 1) % 2];
                        //         }
                        //     }
                        //     for (int i = 0; i < TNS; i++)
                        //     {
                        //         for (int k = 0; k < TNO; k++)
                        //         {
                        //             new_o[i][1][k] = temp_o[i][1][k];
                        //         }
                        //         for (int k = 0; k < TNS; k++)
                        //         {
                        //             new_t[i][1][k] = temp_t[i][1][k];
                        //         }
                        //     }
                        // }
                        if (likelihood(pomdp, new_o, new_t) > likelihood(pomdp, best_o, best_t))
                        {
                            copy_matrix(new_o, best_o);
                            copy_matrix(new_t, best_t);
                            copy_matrix(new_ro, best_ro);
                        }
                    } while (next_permutation(statearray[1], statearray[1] + TNS));        
                } while (next_permutation(statearray[0], statearray[0] + TNS));

                // // TEMPORARY HACK, because the above does not work when triples are composed of identical actions!
                // for (int i = 0; i < TNS; i++)
                // {
                //     for (int k = 0; k < TNO; k++)
                //     {
                //         best_o[i][1][k] = est_o[1 - i][1][k]; 
                //     }
                //     for (int k = 0; k < TNR; k++)
                //     {
                //         best_ro[i][1][k] = est_ro[1 - i][1][k]; 
                //     }
                //     for (int k = 0; k < TNS; k++)
                //     {
                //         best_t[i][1][k] = est_t[1 - i][1][1 - k]; 
                //     }
                // }
                copy_matrix(best_o, est_o);
                copy_matrix(best_t, est_t);
                copy_matrix(best_ro, est_ro);
                cout << "Observation Matrix:" << endl; 
                print_matrix(est_o);
                cout << "Transition Matrix:" << endl;
                print_matrix(est_t);
                cout << "Reward Matrix:" << endl;
                print_matrix(est_ro);
            }

            if (use_mom and update_model)
            {
                for (int i = 0; i < TNS; ++i)
                {
                    for (int j = 0; j < TNA; ++j)
                    {
                        est_r[i][j] = 0;
                        for (int k = 0; k < TNR; ++k)
                        {
                            est_r[i][j] += est_ro[i][j][k]*ro_map[k];
                        }
                        double fake_count = 0;
                        double confidence_alpha = 0.1;
                        for (int ep = 0; ep < pomdp.actions.size(); ep++)
                        {
                            for (int c = 0; c < pomdp.actions[ep].size(); c++)
                            {
                                if (j == pomdp.actions[ep][c]) {fake_count += 1;}
                            }
                        }
                        double ci_radius = fake_count >= 1.0 ? REWARD_GAP*sqrt((0.5/(scale_r * fake_count))*log (2.0/confidence_alpha)) : 1.0;
                        opt_r[i][j] = min(est_r[i][j] + ci_radius, pomdp.rmax);
                    }
                }
            }
            if (0)
            // if (iter % 50 == 0)
            {
                cout << "-------------------- Iteration " << iter << " --------------------" << endl;
                cout << "(s,a,r) = " << pomdp.states[curr_ep].back() << "," << pomdp.actions[curr_ep].back() << "," << pomdp.rewards[curr_ep].back() << endl;
                cout << "Curr Belief -- ";
                print_vector(plan.curr_belief);
                cout << "estimated transitions" << endl;
                print_both(est_t, err_t);
                cout << "optimistic transitions" << endl;
                print_matrix(plan.opt_t);
                cout << "estimated obs" << endl;
                print_both(est_o, err_o);
                cout << "optimistic obs" << endl;
                print_matrix(plan.opt_z);
                cout << "estimated reward obs" << endl;
                print_both(est_ro, err_ro);
                cout << "estimated rewards" << endl;
                print_both(est_r, opt_r);
                cout << "-------------------- Iteration " << iter+1 << " --------------------" << endl;
            }
            // if (iter % 50 == 0)
            // {
            //     print_vector(pomdp.rewards[curr_ep]);
            // }
            // if (iter % 50 == 0)
            // {
            //     cout << "Curr episode: " << curr_ep << endl;
            //     print_both(est_o, err_o);
            //     cout << "---------" << endl;
            //     print_both(est_t, err_t);
            //     cout << "---------" << endl;
            // }
            double recent_reward = pomdp.rewards[curr_ep].back();
            eprs[curr_ep] += recent_reward;
            rs[curr_step] += recent_reward;
            if (curr_ep_step >= max_steps_per_ep)
            // if (abs(recent_reward - 10) < 0.01 or abs(recent_reward+100) < 0.01 or curr_ep_step >= max_steps_per_ep - 1)
            {
                // end of an episode
                ++curr_ep;
                curr_ep_step = 0;
                pomdp.new_episode();
                plan.reset_curr_belief();

            }
            if (curr_ep >= eps)
            {
                break;
            }

        }

        for (size_t ep = 0; ep < pomdp.rewards.size(); ++ep)
        {
            for (size_t i = 0; i < pomdp.rewards[ep].size(); ++i)
            {
                //rs[i] += pomdp.rewards[i];
                sum_rewards += pomdp.rewards[ep][i];
            }
        }
        histogram_r << sum_rewards - prev_sum << endl;
        prev_sum = sum_rewards;

        // show some stats on the run
        if (1)
        {
            // cout << "states" << endl;
            // for (size_t ep = 0; ep < pomdp.states.size(); ++ep)
            // {
            //     for (size_t i = 0; i < pomdp.states[ep].size(); ++i)
            //     {
            //         cout << pomdp.states[ep][i] << " ";
            //     }
            // }
            // cout << endl;
            cout << "actions" << endl;
            output_a << "actions rep " << rep << endl;
            for (size_t ep = 0; ep < pomdp.actions.size(); ++ep)
            {
                for (size_t i = 0; i < pomdp.actions[ep].size(); ++i)
                {
                    cout << pomdp.actions[ep][i] << " ";
                    output_a << pomdp.actions[ep][i] << " ";
                }
            }
            output_a << endl;
            cout << endl;
            cout << "reward obs" << endl;
            // for (size_t ep = 0; ep < pomdp.reward_obs.size(); ++ep)
            // {
            //     for (size_t i = 0; i < pomdp.reward_obs[ep].size(); ++i)
            //     {
            //         cout << pomdp.reward_obs[ep][i] << " ";
            //     }
            // }
            // cout << endl;
            ofstream output_r(reward_out + to_string(rep) + ".txt");
            // cout << "rewards" << endl;
            for (size_t ep = 0; ep < pomdp.rewards.size(); ++ep)
            {
                for (size_t i = 0; i < pomdp.rewards[ep].size(); ++i)
                {
                    // cout << pomdp.rewards[ep][i] << " ";
                    output_r << pomdp.rewards[ep][i] << endl;
                }
            }
            output_r.close();
            // cout << "MoM count: 0: " << momcount[0] << ", 1: " << momcount[1] << endl;
        }
        
        // cout << "Rewards: " << sum_rewards - prev_sum << endl;
        cout << "seed " << seed << endl;
        rept = clock() - rept;
        cout << "---- End Rep: " << ((float) rept)/CLOCKS_PER_SEC << endl << endl;;
    }
    ofstream output_r(reward_out + ".txt");

    // Use when we want to save the reward for each EPISODE
    // for (size_t i = 0; i < eprs.size(); ++i)
    // {
    //     eprs[i] /= reps;
    //     output_r << eprs[i] << endl;
    // }

    // Use when we want to save the reward for each STEP
    for (size_t i = 0; i < rs.size(); ++i)
    {
        rs[i] /= reps;
        output_r << rs[i] << endl;
    }

    output_r << sum_rewards/reps << endl;
    histogram_r.close();
    output_r.close();
    
    if (1)
    {
        //cout << "avg rewards" << setw(4) << endl;
        //for (size_t i = 0; i < rs.size(); ++i)
        //{
            //cout << setw(4) << rs[i] << " ";
        //}
        //cout << endl;
        cout << "avg reward per ep" << endl;
        for (size_t i = 0; i < eprs.size(); ++i)
        {
            cout << fixed << setprecision(2) << setw(8) << eprs[i];
        }
        cout << endl;
    }
    
    cout << "Cumulative reward " << sum_rewards/reps << endl;
}

int main()
{
    //test_random();
    
    //find_planning_params();
    
    //test_em("l2_out.txt", "l2_out_err.txt", "linf_out.txt", "linf_out_err.txt");
    //       OPT?, MoM?
    // test_opt(true, true, "optimistic_rewards_halfx_9s.txt");
    // test_opt(true, false, "tworoom_em_opt_episodic_rewards_75_everystep", "tworoom_em_opt_episodic_cumrewards_75_everystep", "tworoom_em_opt_episodic_actions_75_everystep");
    // test_opt(false, false, "tworoom_em_mean_episodic_rewards_everystep", "tworoom_em_mean_episodic_cumrewards_everystep", "tworoom_em_mean_episodic_actions_everystep");
    // test_opt(false, false, "tworoom_em_egreedy_episodic_rewards", "tworoom_em_egreedy_episodic_cumrewards", "tworoom_em_egreedy_episodic_actions", 50);

    // test_opt(false, true, "mean_rewards_9s_new.txt");
    // test_opt(false, false, "optimal_rewards.txt");
    // test_opt(false, false, "sudo.txt");

    // test_opt(true, false, "tiger_all_opt_10_rewards", "tiger_all_opt_10_cumrewards", "tiger_all_opt_10_actions");
    test_opt(false, false, "tiger_all_mean_rewards", "tiger_all_mean_cumrewards", "tiger_all_mean_actions");
    // test_opt(true, false, "newopt.txt", "newopt2.txt", "newopt3.txt");

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
