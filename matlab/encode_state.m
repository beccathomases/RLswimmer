function s_idx = encode_state(s1, s2, Svals)
%ENCODE_STATE  Map (s1,s2) discrete indices to single state index.
%
%   Svals : vector of allowed levels, e.g. -5:5
%   s1,s2 : elements of Svals
%   s_idx : integer in 1..numel(Svals)^2

    nLevels = numel(Svals);

    i1 = find(Svals == s1, 1);
    i2 = find(Svals == s2, 1);

    if isempty(i1) || isempty(i2)
        error('encode_state: invalid s1/s2 relative to Svals.');
    end

    s_idx = i1 + (i2-1)*nLevels;   % 1..nLevels^2
end
