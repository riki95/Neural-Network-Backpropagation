function [v] = oafsigmoid(x)
v = 1 ./ (1 + exp(-x));
end