#ifndef PLANNING_HPP
#define PLANNING_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>
#include <cstdlib>

#define TIGER_NUMACTIONS 4
#define TIGER_NUMSTATES 2
#define TIGER_NUMOBS 2

using namespace std;

template<class T>
void print_vector(vector<T> const &vec)
{
	for (int i = 0; i < vec.size(); ++i)
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

double sample_unif()
{
	return static_cast<double>(rand()) / RAND_MAX;
}

// gamma(1,1) - from wikipedia
double sample_gamma()
{
	return -log(sample_unif());
}

// found this online as well
// give back a sample uniform dirichlet distribution
void sample_dirichlet(vector<double> &v)
{
	double norm = 0.0;
	for (size_t i = 0; i < v.size(); ++i)
	{
		v[i] = sample_gamma();
		norm += v[i];
	}
	// normalize
	for (size_t i = 0; i < v.size(); ++i)
	{
		v[i] /= norm;
	}
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
template <class M, class T, class Z>
struct Planning
{
	// a belief point will simply be a vector<double>
	vector<vector<double> > beliefs;
	vector<AVector> alphas;
	
	// current belief state
	vector<double> curr_belief;
	
	M &pomdp;
	
	double t[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMSTATES];
	double z[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMOBS];
	
	// keep track of the optimistic instantiation of the model
	double opt_t[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMSTATES];
	double opt_z[TIGER_NUMSTATES][TIGER_NUMACTIONS][TIGER_NUMOBS];
	
	Planning(M &pomdp_)
	: pomdp(pomdp_)
	{
		//double init_alpha_val = pomdp.rmax/(1.0-pomdp.gamma);
		double init_alpha_val = 0;
		// initialize a set of beliefs and alpha vectors
		// probably the alpha vectors will need to be initialized optimistically
		// initialize a uniform belief and optimistic alpha
		//for (int i = 0; i < 10; ++i)
		//{
			//vector<double> tmp_b(pomdp.numstates, 0.0);
			//sample_dirichlet(tmp_b);
			//beliefs.push_back(tmp_b);
			//alphas.push_back(AVector());
			//alphas[i].action = rand() % pomdp.numactions;
			//alphas[i].values = vector<double>(pomdp.numstates, init_alpha_val);
		//}
		
		// use a grid of belief points
		const int num_beliefs = 100;
		for (int i = 0; i <= num_beliefs; ++i)
		{
			vector<double> tmp_b(pomdp.numstates, 0.0);
			tmp_b[0] = static_cast<double>(i) / num_beliefs;
			tmp_b[1] = 1.0-tmp_b[0];
			beliefs.push_back(tmp_b);
			alphas.push_back(AVector());
			int ix = i;
			alphas[ix].action = rand() % pomdp.numactions;
			alphas[ix].values = vector<double>(pomdp.numstates, init_alpha_val);
		}
		
		// current belief is uniform
		curr_belief = vector<double>(pomdp.numstates, 1.0/pomdp.numstates);
	}
	
	
	// once we have the optimistic model, we can do a belief update
	void belief_update()
	{
		// to prevent dividing by zero
		const double smoothing = 0.000001;
		
		vector<double> new_belief = curr_belief;
		
		//cout << "****Before ";
		//print_vector(curr_belief);
		int last_obs = pomdp.obs[pomdp.obs.size()-1];
		int last_action = pomdp.actions[pomdp.actions.size()-1];
		double b_sum = 0.0;
		for (int i = 0; i < curr_belief.size(); ++i)
		{
			double new_b = 0.0;
			for (int j = 0; j < curr_belief.size(); ++j)
			{
				//cout << "*****j=" << j << " " << opt_t[j][last_action][i] << endl;
				new_b += curr_belief[j]*opt_t[j][last_action][i];
			}
			new_belief[i] = opt_z[i][last_action][last_obs]*new_b + smoothing;
			//cout << "*****opt_z " << opt_z[i][last_action][last_obs] << endl;
			//cout << "*****new_b " << new_b << endl;
			b_sum += new_belief[i];
		}
		// normalize
		for (int i = 0; i < curr_belief.size(); ++i)
		{
			new_belief[i] /= b_sum;
		}
		curr_belief = new_belief;
		//cout << "****After ";
		//print_vector(curr_belief);
	}
	
	// optimistic belief point backup, and planning
	// this needs to iterate through all possible new alpha vectors
	// finds the best new alpha for all belief points
	// also finds best action for current belief, and the associated optimistic instantiation
	int backup_plan(T const &tm, T const &tw, Z const &zm, Z const &zw)
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
				policy_eval(i, combo, tmp_values, tm, tw, zm, zw);
				// now update the best alpha vector for all beliefs
				for (int j = 0; j < beliefs.size(); ++j)
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
					transfer_opt();
				}
				else if (tmp_v == curr_v) {
					if (find(best_actions.begin(), best_actions.end(), i) == best_actions.end()) {
						best_actions.push_back(i);
					}
				}
				// only when the last possible combination has been tried, stop
				//print_vector(combo);
				bool stop = true;
				for (int j = 0; j < combo.size(); ++j)
				{
					if (combo[j] != alphas.size()-1)
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
        best_action = best_actions[int(floor(sample_unif()*best_actions.size()))];
		return best_action;
	}
	
	// optimistic one-step policy evaluation
	// given a root action, and an alpha vector for each observation
	// given the means and widths of the confidence intervals
	// return a new alpha vector's values
	// also fills in an optimistic instantiation of the model
	void policy_eval(int action, vector<int> const& combo, vector<double> &values, T const &tm, T const &tw, Z const &zm, Z const &zw)
	{
		// maximization over observations
		assert(values.size() == pomdp.numstates);
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
			values[i] = pomdp.r[i][action] + pomdp.gamma * values[i];
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
		for (int i = 0; i < combo.size(); ++i)
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
		for (int i = 0; i < x.size(); ++i)
		{
			val += x[i] * y[i];
		}
		return val;
	}
	
	// give back the sorted indices only
	void isort(vector<double> const &vals, vector<int> &sorted)
	{
		vector<Tuple<double,int> > tmp_vector;
		for (int i = 0; i < vals.size(); ++i)
		{
			tmp_vector.push_back(Tuple<double,int>(vals[i],i));
		}
		sort(tmp_vector.begin(), tmp_vector.end());
		for (int i = 0; i < vals.size(); ++i)
		{
			sorted[i] = tmp_vector[i].b;
		}
	}
	
	// prints out all the beliefs and associated alphas
	void print_points()
	{
		assert(beliefs.size() == alphas.size());
		for (int i = 0; i < beliefs.size(); ++i)
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
		cout << "Rewards" << endl;
		print_vector(pomdp.rewards);
	}
};

#endif // guard
