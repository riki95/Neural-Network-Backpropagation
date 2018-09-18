function [v] = oaflogistic(x)
v = 1 ./ (1 + exp(-x));
end