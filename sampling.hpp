#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <armadillo>

#include "planning.hpp"

// sample a random permutation
void sample_permutation(int n, vector<int> &out_p)
{
    out_p.resize(n);
    for (int i = 0; i < n; ++i)
    {
        out_p[i] = i;
    }
    // randomly swap with rest
    for (int i = 0; i < n; ++i)
    {
        int j = sample_int(i, n-1);
        if (j > i)
        {
            swap(out_p[i], out_p[j]);
        }
    }
}

// sample instantiation of extreme points from the parameters with cis
template <class T, size_t A, size_t B, size_t C>
void sample_extremes(T const (&est_p)[A][B][C], T const (&err_p)[A][B][C], T (&out_p)[A][B][C])
{
    vector<int> permutation;
    for (int i = 0; i < A; ++i)
    {
        for (int j = 0; j < B; ++j)
        {
            // use a random ordering to get an extreme point here
            sample_permutation(C, permutation);
            
            // first set the elements to the lower bounds to give some leeway
            T lsum = 0;
            for (int k = 0; k < C; ++k)
            {
                T lb = max(0.0, est_p[i][j][k] - err_p[i][j][k]);
                out_p[i][j][k] = lb;
                lsum += lb;
            }
            // then give as much weight as possible to each of the picks in turn
            T leeway = 1 - lsum;
            for (int k = 0; k < C; ++k)
            {
                if (leeway <= 0)
                {
                    break; // done
                }
                int curr_ix = permutation[k];
                T ub = min(1.0, est_p[i][j][curr_ix] + err_p[i][j][curr_ix]);
                T avail = min(leeway, ub - out_p[i][j][curr_ix]);
                out_p[i][j][curr_ix] += avail;
                leeway -= avail;
            }
        }
    }
}

#endif
