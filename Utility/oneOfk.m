function [v] = oneOfk(x,ks)
    v = zeros(1,sum(ks)); %create a vector large sum(ks)
    currentbase = 0;
    for i=1:size(x,2)
        v(currentbase + x(i)) = 1;
        currentbase = currentbase + ks(i);
    end
end