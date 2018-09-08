function [v] = dhtanh(x)
v = 1 - (tanh(x) .^ 2);
end