function [v] = softmax(x)
v = exp(x)/sum(exp(x));
end
