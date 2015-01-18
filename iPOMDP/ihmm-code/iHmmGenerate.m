function [ Y, S, varargout ] = iHmmGenerate(D, T, hypers)
%iHmmGenerate Generates samples from an iHMM.
%
%   [Y,S] = iHmmGenerate(D,T,hypers) generates D samples for an
%   iHMM of length T. The S array contains the hidden states used to
%   generate samples while the Y array contains the observations.
%   
%   hypers is a structure containing the parameters of the iHMM:
%   - for an infinite emission alphabet, using the notation of Beal et al.
%   the iHMM uses hypers are either alpha, beta, gamma, betaE, gammaE.
%   - for a finite emission alphabet, using the HDP-HMM notation, use the
%   hyperparameters gamma, alpha0, H.

S = zeros(D,T);
Y = zeros(D,T);

% Beal et al.'s iHMM generation scheme.
if isfield(hypers, 'betaE')
    
    alpha = hypers.alpha;
    beta = hypers.beta;
    gamma = hypers.gamma;
    betaE = hypers.betaE;
    gammaE = hypers.gammaE;

    N = 0;
    NO = 0;
    
    % Generating state sequence samples.
    for d=1:D
        S(d,1) = 1;

        for t=2:T
            K = size(N,1);
            i = S(d,t-1);

            % First decide whether this is a self-transition, an exisiting
            % transition or an oracle transition.
            dist = N(i, :);
            dist(i) = dist(i) + alpha;
            dist(end+1) = beta;
            dist = dist ./ sum(dist);
            c = 1+sum(rand() > cumsum(dist));

            if c==K+1
                % Call the Oracle.
                dist = NO;
                dist(end+1) = gamma;
                dist = dist ./ sum(dist);
                c = 1+sum(rand() > cumsum(dist));

                if c==K+1
                    NO(c) = 1;
                    N(:,c) = 0;
                    N(c,:) = 0;
                end
            end

            % Update counts.
            N(i,c) = N(i,c) + 1;
            S(d,t) = c;
        end
    end

    if nargout >= 2
        % Generating observation sequence samples.
        M = zeros(size(N,1),1);
        MO = 0;
        for d=1:D
            Y(d,1) = 1;
            M(1,1) = 1;

            for t=2:T
                [K E] = size(M);
                i = S(d,t-1);

                % First decide whether to emit an exisiting symbol or whether to
                % call the oracle.
                dist = M(i,:);
                dist(end+1) = betaE;
                dist = dist ./ sum(dist);
                c = 1+sum(rand() > cumsum(dist));

                if c==E+1
                    % Call the Oracle.
                    dist = MO;
                    dist(end+1) = gammaE;
                    dist = dist ./ sum(dist);
                    c = 1+sum(rand() > cumsum(dist));

                    if c==E+1
                        MO(c) = 1;
                        M(:,c) = 0;
                    end
                end
                % Update counts.
                M(i,c) = M(i,c) + 1;
                Y(d,t) = c;
            end
        end
    end
    
% HDP-HMM generation scheme.
elseif isfield(hypers, 'alpha0')
    
    Betas = cell(D,1);
    gamma = hypers.gamma;
    alpha0 = hypers.alpha0;
    H = hypers.H;
    M = length(H);                  % # of symbols
    for d = 1:D
        %Initialise Beta
        b = betarnd(1, gamma);
        Beta = [b, (1-b)];
        
        K = 1;
        N = zeros(K,K);
        E = zeros(K,M);
        Sd = [1];
        Yd = [1 + sum(rand() > cumsum(H ./ sum(H)))];
        % Start generating the hidden sequence while emitting symbols.
        for t = 2:T
            % Sample next state
            r = alpha0*Beta + [N(Sd(t-1),:) 0];
            r = r ./ sum(r);
            nextState = 1 + sum(rand() > cumsum(r));
            if nextState > K
                N(:,nextState) = 0;                  % We have a new state: augment data structures
                N(nextState,:) = 0;
                E(nextState,:) = 0;
            
                % Extend Beta. Standard stick-breaking construction stuff
                b = randbeta(1, gamma);
                BetaU = Beta(end);
                Beta(end) = b * BetaU;
                Beta(end+1) = (1-b)*BetaU;
            
                K = K + 1;
            end
            N(Sd(t-1), nextState) = N(Sd(t-1), nextState) + 1;
            Sd = [Sd, nextState];
            
            %Generate symbol from nextState
            r = H + E(nextState, :);
            r = r ./ sum(r);
            nextSymbol = 1 + sum(rand() > cumsum(r));
            E(nextState, nextSymbol) = E(nextState, nextSymbol) + 1;
            Yd = [Yd, nextSymbol];
        end
        
        S(d,:) = Sd;
        Y(d,:) = Yd;
        Betas{d} = Beta;
    end
    varargout{1} = Betas;
else
    error('Incorrect number of hyperparameters given');
end
            
        
        
        
    
    