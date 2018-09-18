function [v] = doafsigmoid(x)
v = oafsigmoid(x) .* (1 - oafsigmoid(x)); 
end