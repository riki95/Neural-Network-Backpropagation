function [X_scaled] = scaleInput(X, scale_interval)
%SCALEINPUT scale input between scale_interval(1) and scale_interval(2)

% INPUT
% X: matrix with one pattern per row and one feature per column
% scale_interval: array with two entries: min range and max range
%
% OUTPUT
% X_scaled: matrix with one pattern per row and one feature per column.
%           each value is scaled between scale_interval(1) and scale_interval(2)

% scalars, typically [-1, 1]
min_val = scale_interval(1);
max_val = scale_interval(2);

% min and max of each attribute (row vector)
min_X = min(X);
max_X = max(X);

X_scaled = ((X - min_X) ./ (max_X - min_X)) * (max_val - min_val) + min_val;
end

