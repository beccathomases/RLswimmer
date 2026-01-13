function phi = featurize_state(s_idx, Svals)
%FEATURIZE_STATE  Map discrete state index to feature vector phi.
%
%   Here we just use the discrete indices [s1; s2] as features.
%   You could switch to angles or add nonlinear features later.

    [s1, s2] = decode_state(s_idx, Svals);

    % Option A: raw indices
    phi = [s1; s2];            % 2x1 column vector

    % Option B (commented): use angles instead
    % dtheta = pi/20;
    % theta1 = s1 * dtheta;
    % theta2 = s2 * dtheta;
    % phi = [theta1; theta2];
end
