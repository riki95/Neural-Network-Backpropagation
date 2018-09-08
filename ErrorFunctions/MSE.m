function [v] = MSE(t,nn,reg)
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
    
    diff_target_output = nn.outs{nn.num_hidden+1,1} - t;
    
    v =  0.5 * norm(diff_target_output)^2 + (nn.lambda * tik);
    
end