function [v] = dhrelu_elementwise(x)
    if x <= 0
        v = 0;
    else
        v = 1;
    end
end