function tik = regularizationError(nn)

    % Tikhonov regularizer
    tik = 0;
    
    if reg
        for i=1:size(nn.weights,1)
            tik = tik + sum(sum( nn.weights{i,1} .^ 2));
            
%             % bias weights are not regularized?
%             if nn.bias
%                 biasum = sum(nn.weights{i,1} .^ 2,1);
%                 tik = tik - biasum(size(nn.weights{i,1},2));
%             end
        end
    end
end