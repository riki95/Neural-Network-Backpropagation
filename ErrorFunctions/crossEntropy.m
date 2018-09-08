function v = crossEntropy(t,nn,reg)
    % t = target
    % nn = neural network
    % reg = 1 if Tikhonov has to be applied, 0 otherwise
    
    if nargin < 3
        reg = 0;
    end
    
    tik = 0;
    if reg
        tik = regularizationError(nn);
    end
    
    out = nn.outs{nn.num_hidden+1,1};
    
    v =  sum(-t .* log(out) - (1-t) .* log(1 - out)) + (nn.lambda * tik);
    
end