function v = dcrossEntropy(t,x)
v = (x - t) ./ ((1 - x) .* x);
end