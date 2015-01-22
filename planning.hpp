#ifndef PLANNING_HPP
#define PLANNING_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <random>

#ifndef TIGER_NUMACTIONS
#define TIGER_NUMACTIONS 3
#endif

#ifndef TIGER_NUMSTATES
#define TIGER_NUMSTATES 2
#endif

#ifndef TIGER_NUMOBS
#define TIGER_NUMOBS 2
#endif

#ifndef NUM_BELIEFS
#define NUM_BELIEFS 20
#endif

using namespace std;

mt19937 random_engine;

template<class T>
void print_vector(vector<T> const &vec)
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        cout << vec[i] << " ";
    }
    cout << endl;
}

template<class T>
void print_array(T const &arr, int length)
{
    for (int i = 0; i < length; ++i)
    {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void sample_seed(unsigned int s)
{
    random_engine.seed(s);
}

unsigned int sample_rand()
{
    return random_engine();
}

double sample_unif()
{
    static_assert(random_engine.min() == 0, "Error");
    return static_cast<double>(random_engine()) / random_engine.max();
}

unsigned int sample_int(unsigned int a, unsigned int b)
{
    static_assert(random_engine.min() == 0, "Error");
    unsigned int num = static_cast<unsigned int>(ceil(sample_unif()*(b - a + 1))) + a - 1;
    // static_assert(a <= num <= b);
    return num;
}

template<class T, class R>
struct Tuple
{
    T a;
    R b;
    
    Tuple(T a_, R b_): a(a_), b(b_) {}
    
    bool operator<(Tuple const &rhs) const
    {
        return a < rhs.a;
    }
};

// alpha vector: a vector of values per states, and a root action
struct AVector
{
    int action;
    vector<double> values;
};

// don't know exactly what the types will be yet, so just make it abstract
// M - type of the POMDP
// T - T[][][] transition matrix
// Z - Z[][][] observation matrix
// R - R[][] expected reward matrix
template <class M, class T, class Z, class R>
struct Planning
{
    // a belief point will simply be a vector<double>
    vector<vector<double> > beliefs;
    vector<AVector> alphas;
    
    // current belief state
    vector<double> curr_belief;
    
    M &pomdp;
    
    // temporary variables for calculations
    double t[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMSTATES];
    double z[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMOBS];
    
    // keep track of the optimistic instantiation of the model
    double opt_t[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMSTATES];
    double opt_z[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMOBS];
    
    Planning(M &pomdp_)
    : pomdp(pomdp_)
    {
        //double init_alpha_val = pomdp.rmax/(1.0-pomdp.gamma);
        //double init_alpha_val = 0;
        // initialize a set of beliefs and alpha vectors
        
        // initialize belief points
        reset_belief_points();
        
        // current belief is uniform
        reset_curr_belief();
    }
    
    // void reset_curr_belief()
    // {
    //     curr_belief = vector<double>(pomdp.numstates, 1.0/pomdp.numstates);
    // }

    void reset_curr_belief()
    {
        curr_belief = vector<double>(pomdp.numstates, 0.5);
        // curr_belief[0] = 1.0;
        // curr_belief[1] = 0.0;
    }
    
    void reset_belief_points()
    {
        // use a grid of belief points
        // so far this only works for pomdps with two states
        assert(pomdp.numstates == 2);
        beliefs.clear();
        alphas.clear();
        for (int i = 0; i <= NUM_BELIEFS; ++i)
        {
            vector<double> tmp_b(pomdp.numstates, 0.0);
            tmp_b[0] = static_cast<double>(i) / NUM_BELIEFS;
            tmp_b[1] = 1.0-tmp_b[0];
            beliefs.push_back(tmp_b);
            alphas.push_back(AVector());
            int ix = i;
            alphas[ix].action = sample_int(0, pomdp.numactions-1);
            alphas[ix].values = vector<double>(pomdp.numstates, 0);
        }
    }
    
    // given a new vector of the same size as the beliefs, put the new belief out in there
    void belief_update_step(int last_obs, int last_action, vector<double> const& src_belief, vector<double> &dst_belief)
    {
        //cout << "****Before ";
        //print_vector(src_belief);
        double b_sum = 0.0;
        for (size_t i = 0; i < src_belief.size(); ++i)
        {
            double new_b = 0.0;
            for (size_t j = 0; j < src_belief.size(); ++j)
            {
                //cout << "*****j=" << j << " " << opt_t[j][last_action][i] << endl;
                new_b += src_belief[j]*opt_t[j][last_action][i];
            }
            if (new_b <= pow(10, -200))
            {
                new_b = 0.0;
            }
            dst_belief[i] = opt_z[i][last_action][last_obs]*new_b;
            b_sum += dst_belief[i];
        }
        // cout << b_sum << endl;
        // normalize, but accounting for zero in which case it's uniform
        double factor,smoothing;
        // cout << b_sum << endl;
        assert(b_sum >= 0.0);
        if (b_sum != 0.0)
        {
            // nonzero so just normalize
            factor = 1.0/b_sum;
            smoothing = 0.0;
        }
        else
        {
            // zero, so reset to uniform
            factor = 0.0;
            smoothing = 1.0/src_belief.size();
        }
        for (size_t i = 0; i < src_belief.size(); ++i)
        {
            dst_belief[i] = dst_belief[i] * factor + smoothing;
        }
        //cout << "****After ";
        //print_vector(src_belief);
    }
    
    // once we have the optimistic model, we can do a complete belief update from the beginning
    void belief_update_full()
    {
        int epi = pomdp.epi;
        // reset the current belief to be uniform
        reset_curr_belief();
        vector<double> new_belief(curr_belief.size(), 0.0);
        
        vector<double> *src_belief = &curr_belief;
        vector<double> *dst_belief = &new_belief;
        
        assert(pomdp.obs[epi].size() == pomdp.actions[epi].size());
        // update belief from the very beginning
        for (size_t i = 0; i < pomdp.obs[epi].size(); ++i)
        {
            belief_update_step(pomdp.obs[epi][i], pomdp.actions[epi][i], *src_belief, *dst_belief);
            std::swap(src_belief, dst_belief);
            // src_belief contains the most up to date beliefs at this point
        }
        // copy the updated into curr_belief if not already there
        if (src_belief != &curr_belief)
        {
            curr_belief = *src_belief;
        }
    }
    
    
    // once we have the optimistic model, we can do a belief update for one step
    void belief_update()
    {
        int epi = pomdp.epi;
        vector<double> new_belief(curr_belief.size(), 0.0);
        
        int last_obs = pomdp.obs[epi][pomdp.obs[epi].size()-1];
        int last_action = pomdp.actions[epi][pomdp.actions[epi].size()-1];
        
        belief_update_step(last_obs, last_action, curr_belief, new_belief);
        curr_belief = new_belief;
    }
    
    // optimistic belief point backup, and planning
    // this needs to iterate through all possible new alpha vectors
    // finds the best new alpha for all belief points
    // also finds best action for current belief, and the associated optimistic instantiation
    // so need to assume we have already updated beliefs
    int backup_plan_step(T const &tm, T const &tw, Z const &zm, Z const &zw, R const &er, bool update_opt=false)
    {
        // keep track of the best values for each belief point
        vector<double> vs(beliefs.size(), -numeric_limits<double>::infinity());
        // the new alpha vectors
        vector<AVector> new_alphas = alphas;
        
        // at the same time, find the best new alpha vector and value for the current belief
        double curr_v = -numeric_limits<double>::infinity();
        int best_action = -1;
        vector<int> best_actions(0, 0);
        
        // used to hold a new alpha vector
        vector<double> tmp_values(pomdp.numstates, 0.0);
        
        for (int i = 0; i < pomdp.numactions; ++i)
        {
            vector<int> combo(pomdp.numobs,alphas.size()-1);
            // iterate through all possible new policies
            for(;;)
            {
                // get the next combination
                next_combination(combo, alphas.size());
                // calculate the value of this new alpha vector
                policy_eval(i, combo, tmp_values, tm, tw, zm, zw, er);
                // now update the best alpha vector for all beliefs
                for (size_t j = 0; j < beliefs.size(); ++j)
                {
                    double tmp_v = dot(beliefs[j], tmp_values);
                    if (tmp_v > vs[j])
                    {
                        vs[j] = tmp_v;
                        new_alphas[j].action = i;
                        new_alphas[j].values = tmp_values;
                    }
                }
                // update for current belief
                double tmp_v = dot(curr_belief, tmp_values);
                if (tmp_v > curr_v)
                {
                    curr_v = tmp_v;
                    best_actions.clear();
                    best_actions.push_back(i);
                    // update the optimistic instantiation of the model
                    if (update_opt)
                    {
                        transfer_opt();
                    }
                }
                else if (tmp_v == curr_v) {
                    if (find(best_actions.begin(), best_actions.end(), i) == best_actions.end()) {
                        best_actions.push_back(i);
                    }
                }
                // only when the last possible combination has been tried, stop
                //print_vector(combo);
                bool stop = true;
                for (size_t j = 0; j < combo.size(); ++j)
                {
                    if (combo[j] != (int)alphas.size()-1)
                    {
                        stop = false;
                    }
                }
                //cout << "stop " << stop << endl;
                if (stop)
                {
                    break;
                }
            }
        }
        alphas.swap(new_alphas);
        best_action = best_actions[sample_int(0, best_actions.size()-1)];
        
        // see what actions are actually there
        if (0 and update_opt)
        {
            cout << "best actions ";
            print_vector(best_actions);
        }
        
        return best_action;
    }
    
    // optimistic belief point backup, and planning
    // returns the best action to do
    // does the backup step many times
    int backup_plan(T const &tm, T const &tw, Z const &zm, Z const &zw, R const &er, bool reset=false, int iters=1)
    {
        if (reset)
        {
            reset_belief_points();
        }
        
        for (int i = 0; i < iters-1; ++i)
        {
            backup_plan_step(tm, tw, zm, zw, er);
        }
        return backup_plan_step(tm, tw,zm,zw,er,true);
    }
    
    // optimistic one-step policy evaluation
    // given a root action, and an alpha vector for each observation
    // given the means and widths of the confidence intervals
    // return a new alpha vector's values
    // also fills in an optimistic instantiation of the model
    void policy_eval(int action, vector<int> const& combo, vector<double> &values, T const &tm, T const &tw, Z const &zm, Z const &zw, R const &er)
    {
        // maximization over observations
        assert((int)values.size() == pomdp.numstates);
        // keep track of the inner max alpha values
        vector<double> tilde(pomdp.numstates, 0.0);
        // for loop over s'
        for (int i = 0; i < pomdp.numstates; ++i)
        {
            // initialize the real values to lower bounds first
            for (int j = 0; j < pomdp.numobs; ++j)
            {
                z[i][action][j] = max(zm[i][action][j] - zw[i][action][j], 0.0);
            }
            // << "Lower bounds for action = " << action << " and s' = " << i << endl;
            // print_array(z[i][action], pomdp.numobs);
            // get the alpha_z(s')s over z
            vector<double> zs(pomdp.numobs, 0);
            for (int j = 0; j < pomdp.numobs; ++j)
            {
                zs[j] = alphas[combo[j]].values[i];
            }
            // get the sorted indices
            vector<int> sorted(pomdp.numobs, 0);
            isort(zs, sorted);
            // start from the end and iterate to the beginning of the sorted indices
            for(int ix = pomdp.numobs-1; ix >= 0; --ix)
            {
                int curr_obs = sorted[ix];
                // find the current sum of the obs probs
                double sum_obs = 0.0;
                for (int j = 0; j < pomdp.numobs; ++j)
                {
                    sum_obs += z[i][action][j];
                }
                // if allocated all probs, then stop
                if (sum_obs >= 1.0)
                {
                    break;
                }
                // otherwise, allocate as much as we can
                double gap = zm[i][action][curr_obs] + zw[i][action][curr_obs] - z[i][action][curr_obs];
                z[i][action][curr_obs] += min(1-sum_obs, gap);
            }
            // calculate the dot product
            for (int j = 0; j < pomdp.numobs; ++j)
            {
                // keep updating the alpha vector
                tilde[i] += z[i][action][j] * alphas[combo[j]].values[i];
            }
        }
        
        // now the outer maximization over the transitions
        // for loop over s
        for (int i = 0; i < pomdp.numstates; ++i)
        {
            // initialize the real values to lower bounds first
            for (int j = 0; j < pomdp.numstates; ++j)
            {
                t[i][action][j] = max(tm[i][action][j] - tw[i][action][j], 0.0);
            }
            // get the order of the states such that tilde is in ascending order
            vector<int> sorted(pomdp.numstates, 0);
            isort(tilde, sorted);
            // start from the end and iterate to the beginning of the sorted indices
            for(int ix = pomdp.numstates-1; ix >= 0; --ix)
            {
                int curr_state = sorted[ix];
                // find the current sum of the transition probs
                double sum_tran = 0.0;
                for (int j = 0; j < pomdp.numstates; ++j)
                {
                    sum_tran += t[i][action][j];
                }
                // if allocated all probs, then stop
                if (sum_tran >= 1.0)
                {
                    break;
                }
                // otherwise, allocate as much as we can
                double gap = tm[i][action][curr_state] + tw[i][action][curr_state] - t[i][action][curr_state];
                t[i][action][curr_state] += min(1-sum_tran, gap);
            }
            // keep updating the alpha vector
            values[i] = 0.0;
            for (int j = 0; j < pomdp.numstates; ++j)
            {
                values[i] += t[i][action][j] * tilde[j];
            }
        }
        
        // now we should have an optimistic model
        // now do the final backup step
        for (int i = 0; i < pomdp.numstates; ++i)
        {
            values[i] = er[i][action] + pomdp.gamma * values[i];
        }
    }
    
    // transfers the t,z to opt_t and opt_z
    void transfer_opt()
    {
        for (int i = 0; i < pomdp.numstates; ++i)
        {
            for (int j = 0; j < pomdp.numactions; ++j)
            {
                for (int k = 0; k < pomdp.numstates; ++k)
                {
                    opt_t[i][j][k] = t[i][j][k];
                }
                for (int k = 0; k < pomdp.numobs; ++k)
                {
                    opt_z[i][j][k] = z[i][j][k];
                }
            }
        }
    }
    
    // helper function
    // start with (0,...,0), and keeps advancing to the next possible combination
    // given a particular base i.e. number of possibilities for each component
    void next_combination(vector<int> &combo, int base)
    {
        int carry = 1;
        for (size_t i = 0; i < combo.size(); ++i)
        {
            combo[i] += carry;
            // carry to the next place
            carry = combo[i] / base;
            combo[i] = combo[i] % base;
            if (carry == 0)
            {
                // can stop carrying
                return;
            }
        }
    }
    
    // helper dot product function
    double dot(vector<double> const &x, vector<double> const &y)
    {
        assert(x.size() == y.size());
        double val = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            val += x[i] * y[i];
        }
        return val;
    }
    
    // give back the sorted indices only
    void isort(vector<double> const &vals, vector<int> &sorted)
    {
        vector<Tuple<double,int> > tmp_vector;
        for (size_t i = 0; i < vals.size(); ++i)
        {
            tmp_vector.push_back(Tuple<double,int>(vals[i],i));
        }
        sort(tmp_vector.begin(), tmp_vector.end());
        for (size_t i = 0; i < vals.size(); ++i)
        {
            sorted[i] = tmp_vector[i].b;
        }
    }
    
    // prints out all the beliefs and associated alphas
    void print_points()
    {
        assert(beliefs.size() == alphas.size());
        for (size_t i = 0; i < beliefs.size(); ++i)
        {
            cout << "Belief " << i << " -- ";
            print_vector(beliefs[i]);
            cout << "-- action " << alphas[i].action << " -- ";
            print_vector(alphas[i].values);
        }
    }
    
    // for debugging and testing correctness
    void test()
    {
        vector<int> a(3,0);
        for (int i = 0; i < 10; ++i)
        {
            print_vector(a);
            next_combination(a, 3);
        }
        
        cout << -numeric_limits<double>::infinity() << endl;
        
        vector<double> vals;
        vals.push_back(1);
        vals.push_back(3);
        vals.push_back(4);
        vals.push_back(2);
        vector<int> sorted(vals.size(), 0);
        isort(vals, sorted);
        cout << "Sorted: ";
        print_vector(sorted);
        
        // make up CIs
        double tm[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMSTATES];
        double tw[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMSTATES];
        double zm[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMOBS];
        double zw[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMOBS];
        
        cout << "---------- Start ----------" << endl;
        print_points();
        
        for (int r = 0; r < 200; ++r)
        {
            // update confidence intervals
            for (int i = 0; i < TIGER_NUMSTATES; ++i)
            {
                for (int j = 0; j < TIGER_NUMACTIONS; ++j)
                {
                    for (int k = 0; k < TIGER_NUMSTATES; ++k)
                    {
                        //tm[i][j][k] = 1.0 / TIGER_NUMSTATES;
                        tm[i][j][k] = pomdp.t[i][j][k];
                        tw[i][j][k] = 1.0/sqrt(r);
                        //tw[i][j][k] = 0.1;
                    }
                }
            }
            for (int i = 0; i < TIGER_NUMSTATES; ++i)
            {
                for (int j = 0; j < TIGER_NUMACTIONS; ++j)
                {
                    for (int k = 0; k < TIGER_NUMOBS; ++k)
                    {
                        //zm[i][j][k] = 1.0 / TIGER_NUMOBS;
                        zm[i][j][k] = pomdp.o[i][j][k];
                        zw[i][j][k] = 1.0/sqrt(r);
                        //zw[i][j][k] = 0.1;
                    }
                }
            }
            
            cout << "---------- Iteration " << r+1 << " ----------" << endl;
            cout << "Curr Belief -- ";
            print_vector(curr_belief);
            int next_action = backup_plan(tm, tw, zm, zw);
            cout << "next action: " << next_action << endl;
            
            // advance the pomdp
            pomdp.step(next_action);
            
            // update beliefs
            belief_update();
            
            cout << "opt T" << endl;
            for (int i = 0; i < TIGER_NUMSTATES; ++i)
            {
                for (int j = 0; j < TIGER_NUMACTIONS; ++j)
                {
                    for (int k = 0; k < TIGER_NUMSTATES; ++k)
                    {
                        cout << opt_t[i][j][k] << " ";
                    }
                    cout << "| ";
                }
                cout << endl;
            }
            cout << "opt Z" << endl;
            for (int i = 0; i < TIGER_NUMSTATES; ++i)
            {
                for (int j = 0; j < TIGER_NUMACTIONS; ++j)
                {
                    for (int k = 0; k < TIGER_NUMOBS; ++k)
                    {
                        cout << opt_z[i][j][k] << " ";
                    }
                    cout << "|";
                }
                cout << endl;
            }
            
            print_points();
        }
        // cout << "Rewards" << endl;
        // print_vector(pomdp.rewards);
    }
};

#endif // guard
