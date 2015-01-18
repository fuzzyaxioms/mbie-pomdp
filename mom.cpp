#define  ARMA_DONT_USE_WRAPPER
#include <iostream>
#include <random>
#include <armadillo>
// #include "main2.cpp"
using namespace std;
using namespace arma;

void LearnHMM(mat &O, mat &T, cube x, int k) 
{
	bool fail = false;
	int n = x.n_cols;
	int d = x.n_rows;
	mat P12 = 1.0/n * x.slice(0) * trans(x.slice(1));
	mat P13 = 1.0/n * x.slice(0) * trans(x.slice(2));
	if (arma::rank(P12) < d or arma::rank(P13) < d)
	{
		return;
	}

	cube P123 = zeros<cube>(d, d, d);
	cube P132 = zeros<cube>(d, d, d);
	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < d; j++)
		{
			for (int l = 0; l < d; l++)
			{
				P123(i,j,l) = 1.0/n * sum(x.slice(0).row(i) % x.slice(1).row(j) % x.slice(2).row(l));
				P132(i,j,l) = 1.0/n * sum(x.slice(0).row(i) % x.slice(1).row(l) % x.slice(2).row(j));
			}
		}
	}
	mat U1, U2, V, U3, G, Theta, R;
	vec s;
	svd(U1, s, U2, P12);
	svd(V, s, U3, P13);
	U1 = U1.cols(0, k-1);
	U2 = U2.cols(0, k-1);
	U3 = U3.cols(0, k-1);
	V = V.cols(0, k-1);

	// Sample a uniform random orthogonal matrix using QR magic. (Bartlett decomposition for Stiefel manifolds.)
	G = randn(k, k);
	qr(Theta, R, G);
	// cout << G << endl;
	// Theta.col()
	// cout << R << endl;
	// Theta = G * inv(R);
	// cout << Theta << endl;

	vec eta1 = U3 * trans(Theta.row(0));
	vec eta2 = U2 * trans(Theta.row(0));

	mat P123eta = zeros<mat>(d, d);
	mat P132eta = zeros<mat>(d, d);

	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < d; j++)
		{
			for (int l = 0; l < d; l++)
			{
				P123eta(i,j) +=  as_scalar(P123.slice(l).row(i).col(j) * eta1.row(l));
				P132eta(i,j) +=  as_scalar(P132.slice(l).row(i).col(j) * eta2.row(l));
			}
		}
	}
	mat B123eta = (trans(U1) * (P123eta * U2)) * inv(trans(U1) * (P12 * U2));
	mat B132eta = (trans(V) * (P132eta * U3)) * inv(trans(V) * (P13 * U3));
	cx_vec evalues;
	cx_mat R1;
	vec zerovec = zeros<vec>(k);
	eig_gen(evalues, R1, B123eta);
	vec imagevalues = imag(evalues);
	mat imagR1 = imag(R1);
	if (any(imagevalues) or any(any(imagR1))) 
	{
		return;
	}
	mat R2 = real(R1);
	mat L1 = zeros<mat>(k, k);
	mat L2 = zeros<mat>(k, k);
	L1.row(0) = trans(real(evalues));
	L2.row(0) = trans(diagvec(inv(R2) * (B132eta * R2)));
	for (int r = 1; r < k; r++)
	{
		eta1 = U3 * trans(Theta.row(r));
		eta2 = U2 * trans(Theta.row(r));

		P123eta = zeros<mat>(d, d);
		P132eta = zeros<mat>(d, d);

		for (int i = 0; i < d; i++)
		{
			for (int j = 0; j < d; j++)
			{
				for (int l = 0; l < d; l++)
				{
					P123eta(i,j) +=  as_scalar(P123.slice(l).row(i).col(j) * eta1.row(l));
					P132eta(i,j) +=  as_scalar(P132.slice(l).row(i).col(j) * eta2.row(l));

				}
			}
		}

		B123eta = (trans(U1) * (P123eta * U2)) * inv(trans(U1) * (P12 * U2));
		B132eta = (trans(V) * (P132eta * U3)) * inv(trans(V) * (P13 * U3));
		L1.row(r) = trans(diagvec(inv(R2) * (B123eta * R2)));
		L2.row(r) = trans(diagvec(inv(R2) * (B132eta * R2)));
	}
	O = U2 * (inv(Theta) * L2);

	mat OT = U3 * (inv(Theta) * L1);
	mat norm_O = zeros<mat>(d, k);
	O = max(O, zeros<mat>(d, k));
	rowvec n_O = sum(O, 0);
	for (int i = 0; i < d; i++)
	{
		norm_O.row(i) = n_O;
	}
	O = O/norm_O;
	// Check what this condition is.
	for (int i = 0; i < d; i++)
	{
		if (not any(O.row(i)))
		{
			fail = true; 
		}
	}
	if (arma::rank(O) != k)
	{
		fail = true; 
	}
	if (fail)
	{
		O = zeros<mat>(d, k);
		return;
	}
	T = pinv(O) * OT;
	mat norm_T = zeros<mat>(k, k);
	T = max(T, zeros<mat>(k, k));
	urowvec nonzeros = any(T, 0);
	if (not all(nonzeros))
	{
		O = zeros<mat>(d, k);
		T = zeros<mat>(k, k);
		return;
	}
	rowvec n_T = sum(T, 0);
	for (int i = 0; i < k; i++)
	{
		norm_T.row(i) = n_T;
	}
	T = T/norm_T;
}

// int main()
// {
//   // srand(time(0));
//   sample_seed(49);

//   int d = 6;
//   int lensample = 3;
//   int samplesize = 10000;

//   mat I = eye<mat>(d, d);

//   cube obs = zeros<cube>(d, samplesize, lensample);

//   for (int i = 0; i < samplesize; i++)
//   {
//   	POMDP_OneArm pomdp;
//   	// int a = sample_int(0, 1);
//   	pomdp.step(0);
//   	obs.slice(0).col(i) = I.col(pomdp.obs[0]);
// 	for (int j = 1; j < lensample; j++) {
// 		pomdp.step(0);
// 		obs.slice(j).col(i) = I.col(pomdp.obs[j]);
// 	}
// 	// print_vector(pomdp.obs);

//   }
//   for (int i = 0; i < lensample; i++)
//   {
// 	cout << "[";
// 	for (int j = 0; j < d; j++)
// 	{
// 		cout << "[";
// 		for (int l = 0; l < samplesize; l++)
// 		{
// 			cout << as_scalar(obs.slice(i).row(j).col(l)) << ", ";
// 		}
// 		cout << "], ";
// 	}
// 	cout << "], ";
//   }
//   cout << endl;
//   mat O = zeros<mat>(2, 2);
//   mat T = zeros<mat>(2, 2);
//   LearnHMM(O, T, obs, 2);
//   cout << O << endl;
//   cout << T << endl;
//   return 0;
//  }