clear;
clc;


%%
% This experiment tests the performance of the beam and Gibbs sampler on an
% artifical changepoint detection problem.
T = 2500;                       % Length of HMM
K = 25;                         % # of states
L = 5;                          % # of symbols in emission alphabet

As = 0.99;
An = 0.01;
A = zeros(K,K);
for k=1:K
    if k < K
        A(k,k) = As;
        A(k,k+1) = An;
    else
        A(k,k) = 1;
    end
end
E = sample_dirichlet(0.5*ones(1,L), K);
pi = zeros(K,1); pi(1) = 1;
[Y, STrue] = HmmGenerateData(1,T,pi,A,E);

% Sample states using the iHmm Gibbs sampler.
hypers.alpha0_a = 20;
hypers.alpha0_b = 5;
hypers.gamma_a = 3;
hypers.gamma_b = 1;
hypers.H = ones(1,L);


Gjml = 0;
Bjml = 0;
for f=1:5
    [SG{f} statsG{f}] = iHmmSampleGibbs(Y, hypers, 3000, 1, 1, ceil(rand(1,T) * 20));
    Gjml = Gjml + statsG{f}.jml / 5;
    [SB{f} statsB{f}] = iHmmSampleBeam(Y, hypers, 3000, 1, 1, ceil(rand(1,T) * 20));
    Bjml = Bjml + statsB{f}.jml / 5;
end

figure(1)
title('Changepoint Detection')
plot(1:50:3000, Gjml(1:50:3000), 'r-.', 1:50:3000, Bjml(1:50:3000), 'b-')
legend('Gibbs', 'Beam');
xlabel('Iterations');
ylabel('Likelihood');

%hinton(SampleTransitionMatrix(), 1);
%hinton(SampleEmissionMatrix(, hypers.H), 3);


%%
% This is a little experiment to see how many states an hmm with certain
% parameters approximately has.
z = zeros(20);
for a = 0.5:0.5:10
    for g = 0.5:0.5:10
        hypers.alpha0 = a;
        hypers.gamma = g;
        hypers.H = [1 1 1];
        for i=1:10
            [Y,S] = iHmmGenerate(1, 1000, hypers);
            z(a*2,g*2) = z(a*2,g*2) + max(S);
        end
        z(a*2,g*2) = z(a*2,g*2) / 10;
    end
end
% surf(z)
% xlabel('alpha')
% ylabel('gamma')

contour(0.5:0.5:10, 0.5:0.5:10, z)
xlabel('alpha')
ylabel('gamma')


%%
% This is a reproduction of experiment 1 from the original iHmm paper.
T = 800;                       % Length of HMM
K = 4;                         % # of states
L = 8;                         % # of symbols in emission alphabet

A = [ 0.0 0.5 0.5 0.0;
      0.0 0.0 0.5 0.5;
      0.5 0.0 0.0 0.5;
      0.5 0.5 0.0 0.0 ];
E = [ 1 0 0 0 0 0 1 1;
      1 1 1 0 0 0 0 0;
      0 0 1 1 1 0 0 0;
      0 0 0 0 1 1 1 0 ] / 3;
pi = zeros(K,1); pi(1) = 1;
[Y, STrue] = HmmGenerateData(1,T,pi,A,E);

% Sample states using the iHmm Gibbs sampler.
hypers.alpha0_a = 2;
hypers.alpha0_b = 4;
hypers.gamma_a = 2;
hypers.gamma_b = 1;
% hypers.alpha0 = 0.4;
% hypers.gamma = 3.8;
hypers.H = ones(1,L);

Gjml = 0;
Bjml = 0;
for f=1:5
    [SG{f} statsG{f}] = iHmmSampleGibbs(Y, hypers, 1000, 1, 1, ceil(rand(1,T) * 20));
    Gjml = Gjml + statsG{f}.jml / 5;
    [SB{f} statsB{f}] = iHmmSampleBeam(Y, hypers, 1000, 1, 1, ceil(rand(1,T) * 20));
    Bjml = Bjml + statsB{f}.jml / 5;
end

figure(2)
plot(1:20:1000, Gjml(1:20:1000), 'r-.', 1:20:1000, Bjml(1:20:1000), 'b-')
legend('Gibbs', 'Beam');
xlabel('Iterations');
ylabel('Likelihood');




%%
% This is a reproduction of experiment 2 from the original iHmm paper.
T = 800;                       % Length of HMM
K = 4;                         % # of states
L = 3;                         % # of symbols in emission alphabet

A = [ 0.0 0.5 0.5 0.0;
      0.0 0.0 0.5 0.5;
      0.5 0.0 0.0 0.5;
      0.5 0.5 0.0 0.0 ];
E = [ 0.0 0.0 3.0;
      1.0 1.0 1.0;
      3.0 0.0 0.0;
      0.0 3.0 0.0] / 3;
pi = zeros(K,1); pi(1) = 1;
[Y, STrue] = HmmGenerateData(1,T,pi,A,E);

% Sample states using the iHmm Gibbs sampler.
hypers.alpha0_a = 2;
hypers.alpha0_b = 4;
hypers.gamma_a = 2;
hypers.gamma_b = 1;
hypers.H = ones(1,L);

Gjml = 0;
Bjml = 0;
for f=1:5
    [SG{f} statsG{f}] = iHmmSampleGibbs(Y, hypers, 1000, 1, 1, ceil(rand(1,T) * 20));
    Gjml = Gjml + statsG{f}.jml / 5;
    [SB{f} statsB{f}] = iHmmSampleBeam(Y, hypers, 1000, 1, 1, ceil(rand(1,T) * 20));
    Bjml = Bjml + statsB{f}.jml / 5;
end

figure(3)
plot(1:20:1000, Gjml(1:20:1000), 'r-.', 1:20:1000, Bjml(1:20:1000), 'b-')
legend('Gibbs', 'Beam');
xlabel('Iterations');
ylabel('Likelihood');




%%
% Test the iHMM beam sampler wiht normal output.
T = 400;                         % Length of HMM
K = 2;                          % Number of states
D = 1;                          % Number of samples to generates

A = [0.9 0.1;
     0.1 0.9];
E.mu = [-3.0; 3.0];
E.sigma2 = [0.5; 0.5];
pi = [1.0; zeros(K-1,1)];

% Generate data.
[Y, STrue] = HmmGenerateData(D,T,pi,A,E, 'normal');

% Sample states using the iHmm Gibbs sampler.
tic
hypers.alpha0_a = 1;
hypers.alpha0_b = 10;
hypers.gamma_a = 1;
hypers.gamma_b = 1;
% hypers.alpha0 = 1.0;
% hypers.gamma  = 1.5;
hypers.sigma2 = 1.0;
hypers.mu_0 = 0.0;
hypers.sigma2_0 = 1.0;
[SB, statsB] = iHmmNormalSampleBeam(Y, hypers, 1000, 10, 100, ceil(rand(1,T) * 20));
[SG, statsG] = iHmmNormalSampleGibbs(Y, hypers, 1000, 10, 100, ceil(rand(1,T) * 20));

figure(1)
plot(1:1900, statsB.jml, 'b-', 1:1900, statsG.jml, 'r-');
legend('Beam', 'Gibbs');
