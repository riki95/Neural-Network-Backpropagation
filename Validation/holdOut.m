function [nn, train_acc, test_acc, train_err, test_err, iter, best_var] = holdOut(use, X, y, inp_dim, out_dim, iterations, bias, threshold_grad, tr_perc, test_perc, shuffle)

% use, inp_dim, out_dim, iterations, bias, threshold_grad are hyperparameters of neural network
% X, y are the dataset and target
% tr_perc, test_perc, shuffle specify the training set and test set splitting size. Shuffle is a boolean to choose whether to shuffle or not the dataset while constructing partitions.

% it returns trained (best) neural network, and the values of statistics on train, validation and test set


    % split 50 25 25 randomly
    if nargin == 2
        tr_perc = 0.5;
        test_perc = 0.25;
        shuffle = 1;
    elseif tr_perc + test_perc > 1
        error("Split percentage must sum up at most to one");
    end
    
    [x_train, y_train, x_val, y_val, x_test, y_test] = train_validation_test_split(X, y, tr_perc, test_perc, shuffle);
    
    mb_size = size(x_train,1);
    
    % values for the grid search
    hidden_dim = {[30 30]};
    eta = linspace(0.01, 0.6, 2);
    lambda = [1e-5 1e-4];
    alpha = linspace(0.6, 0.9, 3);
    
    training_iterations = 5;
    
    best_err = inf;
    single_errors = zeros(1,training_iterations);
    
    best_d = 1; % hidden dimension
    best_e = 1; % eta
    best_l = 1; % lambda 
    best_a = 1; % alplha
    best_var = 0;
    
    tot_iter = size(hidden_dim, 1) *  size(eta, 2) * size(lambda, 2) * size(alpha,2);
    iter = 1;
    %I train every model with different hyperparameters, then I evaluate
    %them with the validation set. I find the one with the best accuracy
    %and then I calculate the generalization error
    for d = 1 : size(hidden_dim, 1)
        for e = 1 : size(eta, 2)
            for l = 1 : size(lambda, 2)
                for a = 1 : size(alpha, 2)
                    for i = 1 : training_iterations
                        nn = NeuralNetwork(use,inp_dim, out_dim, hidden_dim{d,:}, iterations, eta(e), lambda(l), alpha(a), bias, threshold_grad, mb_size);
                        nn.fit(x_train, y_train);
                        [~, ~, val_err] = nn.test(x_val,y_val);
                        single_errors(1,i) = val_err;
                    end
                    fprintf("Iteration %d/%d completed.\n",iter,tot_iter);
                    iter = iter + 1;
                    errors = mean(single_errors);
                    
                    % variance of the current model
                    variance = var(single_errors);
                    if (errors < best_err)
                        best_err = errors;
                        best_d = d;
                        best_e = e;
                        best_l = l;
                        best_a = a;
                        best_var = variance;
                    end
                end
            end
        end
    end
   
    % retrain on training set + validation set
    nn = NeuralNetwork(use,inp_dim, out_dim, hidden_dim{best_d,:}, iterations, eta(best_e), lambda(best_l), alpha(best_a), bias, threshold_grad, mb_size);
    [train_acc,test_acc,train_err,test_err,iter] = nn.fit([x_train ; x_val], [y_train ; y_val], x_test, y_test);
end
