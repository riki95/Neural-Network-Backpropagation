% monks attribute (discrete attributes)
% a1 in [1,3] -> 3 bit
% a2 in [1,3] -> 3 bit
% a3 in [1,2] -> 2 bit
% a4 in [1,3] -> 3 bit
% a5 in [1,4] -> 4 bit
% a6 in [1,2] -> 2 bit
% Tot = 17 bit
% class in [0,1] -> 2 bit

addpath(genpath(pwd))
oneOfkConversion = [3 3 2 3 4 2];

% prepare training data
data = csvread('Data/Classification/monks-1.csv');
x = data(:,[2 3 4 5 6 7]);
y = data(:,1); 

X = zeros(size(x,1),sum(oneOfkConversion));
for i=1:size(x,1)
    X(i,:) = oneOfk(x(i,:), oneOfkConversion);
end

input_dim = 17; % 3 + 3 + 2 + 3 + 4 + 2
output_dim = 1; 
iterations = 400;
bias = 1;
threshold_grad = 1e-8;
mb_size = 32; %used only with no validation. Otherwise we initialize it in kfold-holdout
shuffle = 1;

validation = 1;
use = 0; % 1 = regression, 0 = classification
assessment = 0;

if ~validation
    
    tr_perc = 0.6;
    test_perc = 0.2;
    
    [X_train,y_train,X_val,y_val,X_test,y_test] = train_validation_test_split(X,y,tr_perc,test_perc,shuffle);
    
    % define hyperparameters
    hidden_dim = 10;
    eta = 0.9; % learning rate
    lambda = 1e-3; % Tykhonov
    alpha = 0.8; % momentum
    
    if mb_size > size(X_train,1)
        error('mb_size larger than number of patterns: %d', size(X,1));
    end
    
    % train neural network
    nn = NeuralNetwork(use,input_dim,output_dim,hidden_dim,iterations,eta,lambda,alpha,bias,threshold_grad,mb_size);

    fprintf('Starting fitting...');
    [train_acc,test_acc,train_err,test_err,iter] = nn.fit(X_train,y_train,X_val,y_val);
    printf;
    
    % test with trained neural network
    [accuracy, outputs, errors_test] = nn.test(X_test,y_test);
    fprintf('Test Accuracy: %f\n', accuracy);
    
else
    % tr_perc and test_perc must sum to 1 only for K-fold CV
    tr_perc = 0.8;
    test_perc = 0.2;
    fold = 5;
    [nn,train_acc, test_acc, train_err, test_err, best_err, iter, best_var] = KFold(assessment,use,X,y,fold,input_dim,output_dim,iterations,bias,threshold_grad,tr_perc,test_perc,shuffle);
    fprintf('Test accuracy: %f\n', test_acc(end));
    fprintf('Test error: %f\n', test_err(end));
    fprintf("Eta: %f\nLambda: %f\nAlpha: %f\n Hidden dimensions: %f %f\n",nn.eta,nn.lambda,nn.alpha, nn.hidden_dim);
    
end

plot_curve(iter, train_err, test_err, train_acc, test_acc);