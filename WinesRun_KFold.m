%Here we have a Regression dataset in which the y column is the last one
%and the first 11 are dimensions. We do not have to recompute the data like
%we did in Monks, we just pass it to the NN as it is.

addpath(genpath(pwd))

%data = dlmread('Data/Regression/winequality-red-ok.csv',';');
data = dlmread('Data/Regression/winequality-white.csv',';');

X = data(:,[1 2 3 4 5 6 7 8 9 10 11]);
y = data(:,12);

% scale between -1 and 1
min_scale = -1;
max_scale = 1;
X = scaleInput(X,[min_scale,max_scale]);

input_dim = 11;
output_dim = 1;
iterations = 50;
bias = 1;
threshold_grad = 1e-2;
shuffle = 1;

validation = 1;
use = 1;
assessment = 0;

if validation == 0
    tr_perc = 0.6;
    test_perc = 0.2;
    [X_train,y_train,X_val,y_val,X_test,y_test] = train_validation_test_split(X,y,tr_perc,test_perc,shuffle);
    
    % define hyperparameters
    hidden_dim = 20;
    eta = 0.01; % learning rate
    lambda = 0.01; % Tykhonov
    alpha = 0.5; % momentum
    mb_size = size(X_train,1);
    
    if mb_size > size(X_train,1)
        error('mb_size larger than number of patterns: %d', size(X_train,1));
    end

    % train
    nn = NeuralNetwork(use,input_dim,output_dim,hidden_dim,iterations,eta,lambda,alpha,bias,threshold_grad,mb_size);
    [~,~,train_err, test_err, iter] = nn.fit(X_train,y_train,X_val,y_val);

    % test
    [~,out,test_error] = nn.test(X_test,y_test);
    fprintf("Test error: %d\n", test_error);
    
else
    % tr_perc and test_perc must sum to 1 only for K-fold CV
    tr_perc = 0.8;
    test_perc = 0.2;
    fold = 5;
    [nn,train_acc, test_acc, train_err, test_err, best_err, iter, var] = KFold(assessment,use,X,y,fold,input_dim,output_dim,iterations,bias,threshold_grad,tr_perc,test_perc,shuffle);    
    fprintf("Training error: %d\n", train_err(end));
    fprintf("Test error: %d\n", test_err(end));
    fprintf("Variance: %d\n", var);
    fprintf("Eta: %f\nLambda: %f\nAlpha: %f\nHidden dim: %f\n",nn.eta,nn.lambda,nn.alpha, nn.hidden_dim);
end

plot_curve(iter, train_err, test_err);