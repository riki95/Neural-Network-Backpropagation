function [v] = doaflogistic(x)
v = oaflogistic(x) .* (1 - oaflogistic(x)); 
end