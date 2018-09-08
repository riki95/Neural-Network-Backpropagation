function [v] = dhrelu(x)
v = arrayfun(@dhrelu_elementwise, x);
end