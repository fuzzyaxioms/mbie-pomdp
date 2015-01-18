% Test the iHMM Beam sampler on the expansive HMM experiment from the
% original iHMM paper [Beal et al. 2002]. The plotted transition and
% emission matrices should be equivalent up to permutation.

disp('Train iHMM test ... (press any key to continue)')
pause

T = 500;                       % Length of HMM
K = 4;                         % # of states
L = 8;                         % # of symbols in emission alphabet

rand('seed', 21);

% Parameters for the HMM that generates the data.
A = [ 0.0 0.5 0.5 0.0;
      0.0 0.0 0.5 0.5;
      0.5 0.0 0.0 0.5;
      0.5 0.5 0.0 0.0 ];
E = [ 1 0 0 0 0 0 1 1;
      1 1 1 0 0 0 0 0;
      0 0 1 1 1 0 0 0;
      0 0 0 0 1 1 1 0 ] / 3;
pi = [1.0; zeros(K-1,1)];

% Generate data.
[Y, STrue] = HmmGenerateData(1,T,pi,A,E);

% Sample states using the iHmm Gibbs sampler.
hypers.alpha0_a = 4;
hypers.alpha0_b = 2;
hypers.gamma_a = 3;
hypers.gamma_b = 6;
hypers.H = ones(1,L) * 0.3;
tic
[S stats] = iHmmSampler(Y, hypers, 500, 1, 1);
toc

% Plot some stats
figure(2)
subplot(3,2,1)
plot(stats.K)
title('K')
subplot(3,2,2)
plot(stats.jll)
title('Joint Log Likelihood')
subplot(3,2,3)
plot(stats.alpha0)
title('alpha0')
subplot(3,2,4)
plot(stats.gamma)
title('gamma')
subplot(3,2,5)
imagesc(S{1}.Pi(:,1:end-1)); colormap('Gray');
title('Transition Matrix')
subplot(3,2,6)
imagesc(S{1}.Phi); colormap('Gray');
title('Emission Matrix')



%% Generate data from an IO-HMM and try to learn an iHMM.
disp('Train IO-iHMM test ... (press any key to continue)')
pause

T = 1000;                      % Length of HMM

rand('seed', 21);

% Parameters for the HMM that generates the data.
A = {[ 0.0 0.9 0.1 0.0;
       0.0 0.0 0.9 0.1;
       0.1 0.0 0.0 0.9;
       0.9 0.1 0.0 0.0 ];
     [ 0.9 0.1 0.0 0.0;
       0.1 0.9 0.0 0.0;
       0.0 0.0 0.9 0.1;
       0.0 0.0 0.1 0.9 ]};
  
E = {[ 1 0 0 0 0 0 1 1;
       1 1 1 0 0 0 0 0;
       0 0 1 1 1 0 0 0;
       0 0 0 0 1 1 1 0 ] / 3;
     [ 1 1 1 0 0 0 0 0;
       0 1 1 1 0 0 0 0;
       0 0 0 0 1 2 0 0;
       0 0 0 0 0 0 1 2 ] / 3};
K = size(A{1},1);                         % # of states
L = size(E{1},2);                         % # of symbols in emission alphabet
pi = {[1.0; zeros(K-1,1)]; [1.0; zeros(K-1,1)]};
obs.M = 2;
obs.I = { repmat([ones(1,50) 2*ones(1,50)], 1, T / 100) };

% Generate data.
[obs.Y, STrue] = IOHmmGenerateData(obs.I,pi,A,E);

% Sample states using the beam sampler.
hypers.alpha0_a = 4;
hypers.alpha0_b = 2;
hypers.gamma_a = 3;
hypers.gamma_b = 6;
hypers.H = ones(1,L) * 0.3;
tic
[S stats] = iHmmSampler(obs, hypers, 500, 1, 1);
toc

% Plot some stats
figure(1)
subplot(3,2,1)
plot(stats.K)
title('K')
subplot(3,2,2)
plot(stats.jll)
title('Joint Log Likelihood')
subplot(3,2,3)
plot(stats.alpha0)
title('alpha0')
subplot(3,2,4)
plot(stats.gamma)
title('gamma')

figure(2);
subplot(2,2,1);
imagesc(S{1}.Phi(:,:,1)); colormap('Gray');
title('Found 1');
subplot(2,2,2);
imagesc(S{1}.Phi(:,:,2)); colormap('Gray');
title('Found 2');

subplot(2,2,3);
imagesc(E{1}); colormap('Gray');
title('True 1');
subplot(2,2,4);
imagesc(E{2}); colormap('Gray');
title('True 2');

figure(3);
subplot(2,1,1);
imagesc(S{1}.S{1});
subplot(2,1,2);
imagesc(STrue{1});