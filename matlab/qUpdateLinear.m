function [W, b] = qUpdateLinear(W, b, phi, a_idx, r, phi_next, terminal, alpha, gamma)
%QUPDATELINEAR  Q-learning update for a linear Q network.
%
%   W, b      : parameters of linear Q-net.
%               Q(s, a) = W(a,:) * phi + b(a)
%               W is (nActions x state_dim), b is (nActions x 1)
%   phi       : current state features (state_dim x 1)
%   a_idx     : action index (1..nActions)
%   r         : reward
%   phi_next  : next state features (state_dim x 1)
%   terminal  : logical; if true, episode terminates after this step
%   alpha     : learning rate
%   gamma     : discount factor

    % current Q(s,a)
    q_sa = W(a_idx,:) * phi + b(a_idx);

    % target
    if terminal
        target = r;
    else
        q_next_all = W * phi_next + b;   % (nActions x 1)
        target = r + gamma * max(q_next_all);
    end

    td_error = target - q_sa;

    % gradient step on W(a_idx,:) and b(a_idx)
    W(a_idx,:) = W(a_idx,:) + alpha * td_error * (phi.');
    b(a_idx)   = b(a_idx)   + alpha * td_error;
end
