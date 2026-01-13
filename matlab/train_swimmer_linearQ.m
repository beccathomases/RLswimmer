function returns = train_swimmer_linearQ(nEpisodes, Tmax, doPlot)
%TRAIN_SWIMMER_LINEARQ  Linear Q-network on 2-paddle swimmer.
%
%   returns = train_swimmer_linearQ(nEpisodes, Tmax, doPlot)
%
%   nEpisodes : number of episodes
%   Tmax      : steps per episode
%   doPlot    : if true, plots cumulative reward vs episode
%
%   returns   : 1 x nEpisodes vector of cumulative reward

    if nargin < 1, nEpisodes = 500; end
    if nargin < 2, Tmax       = 40;  end
    if nargin < 3, doPlot     = false; end

    % ----- state space -----
    Svals    = -5:5;
    nLevels  = numel(Svals);
    nStates  = nLevels^2; %#ok<NASGU> % not needed directly, but nice to know

    % ----- action set for 2 paddles -----
    A = [ -1  0;   % move paddle 1 down
           1  0;   % move paddle 1 up
           0 -1;   % move paddle 2 down
           0  1;   % move paddle 2 up
          -1  1;   % counter-rotate
           1 -1;   % counter-rotate
          -1 -1;   % both down
           1  1 ]; % both up
    nActions = size(A,1);

    % ----- linear Q-network parameters -----
    state_dim = 2;          % [s1; s2] features

    W = zeros(nActions, state_dim);
    b = zeros(nActions, 1);

    % hyperparameters
    alpha     = 0.01;   % smaller step since function approx can be volatile
    gamma     = 0.95;
    eps       = 0.2;
    eps_min   = 0.05;
    eps_decay = 0.995;

    s0 = encode_state(0, 0, Svals);

    returns = zeros(1, nEpisodes);

    for ep = 1:nEpisodes
        s        = s0;
        G        = 0;
        terminal = false;

        for t = 1:Tmax
            % --- featurize current state ---
            phi = featurize_state(s, Svals);  % state_dim x 1

            % --- ε-greedy using linear Q-net ---
            qvals = W * phi + b;              % nActions x 1

            if rand < eps
                a_idx = randi(nActions);
            else
                [~, a_idx] = max(qvals);
            end

            a_vec = A(a_idx,:);

            % environment step
            [s_next, r] = swimmer_step(s, a_vec, Svals);

            terminal = (t == Tmax);

            % featurize next state
            phi_next = featurize_state(s_next, Svals);

            % Q-learning update on W,b
            [W, b] = qUpdateLinear(W, b, phi, a_idx, r, phi_next, terminal, alpha, gamma);

            G = G + r;
            s = s_next;
        end

        returns(ep) = G;
        eps = max(eps_min, eps * eps_decay);

        if mod(ep, 50) == 0
            fprintf('LinearQ ep %4d: G = %.3f, eps = %.3f\n', ep, G, eps);
        end
    end

    if doPlot
        figure; plot(1:nEpisodes, returns, '-'); xlabel('Episode');
        ylabel('Return (cumulative reward)');
        title('Linear Q-network returns');
    end
end
