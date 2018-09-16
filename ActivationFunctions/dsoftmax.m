function [v] = dsoftmax(x)
v = x*(-x +1);
end
