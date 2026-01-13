function [s_next, r] = swimmer_step(s_idx, a_vec, Svals)
%SWIMMER_STEP  One environment step for 2-paddle swimmer (placeholder).
%
%   s_idx : current discrete state index (1..nStates)
%   a_vec : 1x2 action [d1 d2] in {-1,0,1}^2 (row of A)
%   Svals : vector of allowed indices, e.g., -5:5
%
% Returns:
%   s_next : next state index
%   r      : reward (placeholder physics)

    % decode to (s1,s2)
    [s1, s2] = decode_state(s_idx, Svals);

    % apply action in index space with clamping
    Smax   = max(abs(Svals));
    s1_new = max(-Smax, min(Smax, s1 + a_vec(1)));
    s2_new = max(-Smax, min(Smax, s2 + a_vec(2)));

    % convert to angles for the physics model
    dtheta = pi/20;
    theta1 = s1_new * dtheta;
    theta2 = s2_new * dtheta;

    % ----- placeholder reward (toy model) -----
    phase_diff    = theta2 - theta1;
    forward_proxy = 0.1 * sin(phase_diff);
    angle_penalty = 0.01 * (theta1^2 + theta2^2);

    r = forward_proxy - angle_penalty;

    % re-encode next state
    s_next = encode_state(s1_new, s2_new, Svals);
end
