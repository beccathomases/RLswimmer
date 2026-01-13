function returns = train_swimmer_tabular(nEpisodes, Tmax, doPlot)
%TRAIN_SWIMMER_TABULAR  Tabular Q-learning on 2-paddle swimmer.
%
%   returns = train_swimmer_tabular(nEpisodes, Tmax, doPlot)
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
    Svals    = -5:5;                % discrete index levels
    nLevels  = numel(Svals);
    nStates  = nLevels^2;           % (s1,s2) grid

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

    % ----- Q-table -----
    Q = zeros(nStates, nActions);

    % hyperparameters
    alpha    = 0.1;    % learning rate
    gamma    = 0.95;   % discount
    eps      = 0.2;    % initial epsilon
    eps_min  = 0.05;
    eps_decay = 0.995; % per-episode decay

    % start in the "center" configuration
    s0 = encode_state(0, 0, Svals);

    returns = zeros(1, nEpisodes);

    for ep = 1:nEpisodes
        s         = s0;
        G         = 0;    % cumulative reward
        terminal  = false;

        for t = 1:Tmax
            % ε-greedy action selection
            if rand < eps
                a_idx = randi(nActions);           % explore
            else
                [~, a_idx] = max(Q(s,:));          % exploit
            end

            a_vec = A(a_idx,:);

            % step environment
            [s_next, r] = swimmer_step(s, a_vec, Svals);

            % here we treat the horizon as the only terminal condition
            terminal = (t == Tmax);

            % tabular Q update
            if terminal
                target = r;
            else
                target = r + gamma * max(Q(s_next,:));
            end
            Q(s,a_idx) = Q(s,a_idx) + alpha * (target - Q(s,a_idx));

            G = G + r;
            s = s_next;
        end

        returns(ep) = G;
        eps = max(eps_min, eps * eps_decay);

        if mod(ep, 50) == 0
            fprintf('Tabular ep %4d: G = %.3f, eps = %.3f\n', ep, G, eps);
        end
    end

    if doPlot
        figure; plot(1:nEpisodes, returns, '-'); xlabel('Episode');
        ylabel('Return (cumulative reward)');
        title('Tabular Q-learning returns');
    end
end
