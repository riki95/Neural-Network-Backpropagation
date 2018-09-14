function [xTraining, yTraining, xValidation, yValidation, xTest, yTest] = train_validation_test_split(X, y, tr_perc, test_perc, shuffle)
    % tr_perc e test_perc are % of slit data
    % If they sum to 1, validation is empty (used in K-Fold)
    % Shuffle is a bool for random permutation of dataset before the split
    
    N = size(X,1);
    
    if shuffle
        idx = randperm(N);
        X = X(idx,:);
        y = y(idx,:);
    end

    tr_end = round(N * tr_perc);
    test_end = tr_end + round(N * test_perc);

    xTraining = X(1:tr_end,:);
    yTraining = y(1:tr_end,:);
    xTest = X(tr_end+1:test_end,:);
    yTest = y(tr_end+1:test_end,:);
    
    if tr_perc + test_perc == 1 % This is KFold case
        xValidation = [];
        yValidation = [];
    else
        xValidation = X(test_end+1:end,:);
        yValidation = y(test_end+1:end,:);
    end
end