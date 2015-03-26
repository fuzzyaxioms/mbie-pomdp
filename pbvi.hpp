#ifndef PBVI_HPP
#define PBVI_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <tuple>
#include <armadillo>

using namespace std;
using namespace arma;

// represent the transition matrix as T(s,s',a)
// obs matrix as Z(s',z,a)
// reward as r=R(s,a)

struct PBVI
{
	// a belief point is a column vector, so a finite set of beliefs is a matrix
    mat beliefs;
	
	// finite set of alpha vectors as columns
	mat alphas;
	
	// the root action associated with each alpha vector
	uvec root_actions;
	
	double gamma;
	
    void reset_belief_points()
    {
        // use a grid of belief points
        for (uword i = 0; i <= beliefs.n_cols-1; ++i)
        {
			double p = static_cast<double>(i) / (beliefs.n_cols - 1);
			beliefs(0, i) = p;
			beliefs(1, i) = 1.0-p;
        }
    }
	
	void reset_alphas()
	{
		// just set all alphas to zero, which is the same as 0-th alpha step
		alphas.zeros();
	}
	
	void update_alphas_step(cube const & trans, cube const & obs, mat const & rewards, field<vec> & giza, field<vec> & gia)
	{
		// first construct gamma_{a,z} i.e. the new alpha part for every action and observation
		// giza is (i, z,a) where i is the i-th alpha vector, of pre-alpha vectors
		// for alpha_i, z, a, the new alpha is gamma * T(:,:,a) * (Z(:,z,a) .* alpha)
		for (uword curr_a = 0; curr_a < obs.n_slices; ++curr_a)
		{
			mat const & trans_a = trans.slice(curr_a);
			mat const & obs_a = obs.slice(curr_a);
			for (uword curr_z = 0; curr_z < obs.n_cols; ++curr_z)
			{
				vec const & obs_za = obs_a.col(curr_z);
				for (uword curr_i = 0; curr_i < alphas.n_cols; ++curr_i)
				{
					giza(curr_i, curr_z, curr_a) = gamma * (trans_a * (obs_za % alphas.col(curr_i)));
				}
			}
		}
		
		// next we find the best set of alphas to pick for each z for each belief
		// gia is (i,a), of the best alpha that uses action a for i-th belief point
		for (uword curr_i = 0; curr_i < alphas.n_cols; ++curr_i)
		{
			vec const & curr_belief = beliefs.col(curr_i);
			for (uword curr_a = 0; curr_a < obs.n_slices; ++curr_a)
			{
				// start with the immediate rewards
				gia(curr_i, curr_a) = rewards.col(curr_a);
				
				// aggregate the best alpha for each obs
				for (uword curr_z = 0; curr_z < obs.n_cols; ++curr_z)
				{
					double best_alpha_val = numeric_limits<double>::lowest();
					uword best_i = 0;
					
					// find next best alpha for this obs,action combination
					for (uword next_i = 0; next_i < alphas.n_cols; ++next_i)
					{
						double next_val = dot(giza(next_i, curr_z, curr_a), curr_belief);
						if (next_val > best_alpha_val)
						{
							best_alpha_val = next_val;
							best_i = next_i;
						}
					}
					// add this to the aggregate
					gia(curr_i, curr_a) += giza(best_i, curr_z, curr_a);
				}
			}
		}
		
		// finally we find the best action alpha for each belief
		for (uword curr_i = 0; curr_i < alphas.n_cols; ++curr_i)
		{
			vec const & curr_belief = beliefs.col(curr_i);
			
			double best_alpha_val = numeric_limits<double>::lowest();
			uword best_a = 0;
			
			for (uword curr_a = 0; curr_a < obs.n_slices; ++curr_a)
			{
				double next_val = dot(gia(curr_i, curr_a), curr_belief);
				if (next_val > best_alpha_val)
				{
					best_alpha_val = next_val;
					best_a = curr_a;
				}
			}
			// store the best alpha and best a
			alphas.col(curr_i) = gia(curr_i, best_a);
			root_actions(curr_i) = best_a;
		}
	}
	
public:
	template<class M>
	PBVI(M &pomdp, int nbeliefs)
    : beliefs(pomdp.numstates, nbeliefs, fill::zeros),
	  root_actions(nbeliefs, fill::zeros),
	  alphas(pomdp.numstates, nbeliefs, fill::zeros),
	  gamma(pomdp.gamma)
    {
        //double init_alpha_val = pomdp.rmax/(1.0-pomdp.gamma);
        //double init_alpha_val = 0;
        // initialize a set of beliefs and alpha vectors
        
        // initialize belief points
		assert(pomdp.numstates == 2); // currently only works for 2 states
        reset_belief_points();
		reset_alphas();
		// at this point, the root_actions mean nothing
    }
	
	uword num_beliefs() const
	{
		return beliefs.n_cols;
	}
	
	void plan(cube const & trans, cube const & obs, mat const & rewards, uword iters)
	{
		// reset to zero
		reset_alphas();
		
		field<vec> gza(alphas.n_cols, obs.n_cols, obs.n_slices);
		field<vec> gia(alphas.n_cols, obs.n_slices);
		
		for (uword i = 0; i < iters; ++i)
		{
			update_alphas_step(trans, obs, rewards, gza, gia);
		}
	}
	
	tuple<uword,double> find_action(vec const & curr_belief)
	{
		// find the best associated alpha vector and return its action
		vec vals = alphas.t() * curr_belief;
		uword best_i;
		double best_val = vals.max(best_i);
		return make_tuple(root_actions(best_i), best_val);
	}
	
	void debug_print_points()
	{
		cout << join_horiz(join_horiz(conv_to<mat>::from(root_actions), beliefs.t()), alphas.t()) << "\n";
	}
};


#endif // guard
