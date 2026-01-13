function [s1, s2] = decode_state(s_idx, Svals)
%DECODE_STATE  Inverse of encode_state.
%
%   s_idx : integer in 1..numel(Svals)^2
%   Svals : vector of allowed levels, e.g., -5:5
%   s1,s2 : elements of Svals

    nLevels = numel(Svals);

    if s_idx < 1 || s_idx > nLevels^2
        error('decode_state: s_idx out of range.');
    end

    % base-11 style decode
    i2 = floor((s_idx-1)/nLevels) + 1;
    i1 = s_idx - (i2-1)*nLevels;

    s1 = Svals(i1);
    s2 = Svals(i2);
end
