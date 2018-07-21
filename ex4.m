%% PBL Project HSI Fort Worth
% Created by Omer Can

%% Initialization
clear ; close all; clc

%% Hard code some parameters
input_layer_size  = 400;    % 20x20 Input Images of Digits
hidden_layer_size = 25;     % 25 hidden units
num_labels = 10;            % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

set(0, 'DefaultFigureWindowStyle','docked'); % Docked image

numOfIter = 50;             % Change this number but avoid from overfitting
finalLambda = 0.3;            % Change this value to see its effect

%% =========== Part 1: Loading and Visualizing Data =============
% This part is load the sample set data we get from the MNIST database.
% Link: yann.lecun.com/exdb/mnist
% I only use 5000 examples

% Load Training Data
fprintf('=======================================\n');
fprintf('\nLoading and Visualizing Data ...\n')
load('lib\ex4data1.mat');
m = size(X, 1);

% Randomly select and subtract 100 training examples to display and to test later
 numberOfTestImages = 500;
 sel = randperm(size(X, 1));
 test_set = X(sel(1:numberOfTestImages), :);
 test_set_ans = y(sel(1:numberOfTestImages), :);
 displayData(test_set);
 X = X(sel(numberOfTestImages + 1:end), :);
 y = y(sel(numberOfTestImages + 1:end), :);

%% ================ Part 2: Loading Parameters ================
% In this part, I load some pre-initialized 
% neural network parameters so we can check if we are on the right path
% later.

fprintf('=======================================\n');
fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% We need to unroll the Theta values (aka parameters) because passing a
% vector to a function is faster. Therefore, most of the build-in functions
% in MATLAB are using vector as an input. We later use fminunc, which takes parameter as a vector, to minimize
% the cost function.
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Activation Function (aka Sigmoid Function) ================
% Sigmoid function is a useful activation function but there are other
% function that can be used as a activation function. In this part, we
% implement sigmoid function in sigmoid.m file.

fprintf('=======================================\n');
fprintf('\nChecking Sigmoid Function ...\n');

g = sigmoid([-10 -5 -1 0 1 5 10]);

fprintf('Sigmoid Function evaluated at [-10 -5 -1 0 1 5 10]:\n  ');
fprintf('%f ', g);
fprintf('\nThis values should be about:\n');
fprintf('  0.000045 0.006693 0.268941 0.500000 0.731059 0.993307 0.999955\n');
fprintf('Press enter to continue.\n');
pause;

%% ================ Part 4: Compute Cost (Feedforward) ================
% First, we should start by implementing the feedforward part of the NN
% in nnCostFunction.m that returns the cost only so we understand 
% if our feedforward code is right by comparing the pre-calculated cost
% and the cost we find.

fprintf('=======================================\n');
fprintf('\nChecking Cost Function ...\n')

% We set this 0 so we can determine if the cost function without regularization
% is right.
lambda = 0;

J = nnCostFunction(nn_params,...
        input_layer_size,...
        hidden_layer_size,...
        num_labels,...
        X,...
        y,...
        lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287433). Press enter to continue.\n'], J);
pause;

%% =============== Part 5: Implement Regularization ===============
% You will implement the regularization.

fprintf('=======================================\n');
fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% We set this 1 so we can check the regularization term.
lambda = 1;

J = nnCostFunction(nn_params,...
        input_layer_size,...
        hidden_layer_size, ...
        num_labels,...
        X,...
        y,...
        lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.394256). Press enter to continue.\n'], J);
pause;


%% ================ Part 6: Sigmoid Gradient  ================
% Before you start implementing the neural network, you will first need to
% implement the gradient for the sigmoid function. You should complete the
% code in the sigmoidGradient.m file.

fprintf('=======================================\n');
fprintf('\nChecking Sigmoid Gradient...\n')

g = sigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\nThis values should be about:\n');
fprintf('  0.196612 0.235004 0.250000 0.235004 0.196612\n Press enter to continue.\n');
pause;


%% ================ Part 7: Initializing Pameters ================
% In this part of the exercise, you will be starting to implment a two
% layer neural network that classifies digits. You will start by
% implementing a function to initialize the weights of the neural network
% (randInitializeWeights.m)

fprintf('=======================================\n');
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =============== Part 8: Implement Backpropagation ===============
% Once your cost matches up with ours, you should proceed to implement the
% backpropagation algorithm for the neural network. You should add to the
% code you've written in nnCostFunction.m to return the partial
% derivatives of the parameters.
%

fprintf('=======================================\n');
fprintf('\nChecking Backpropagation... \n');

% Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 9: Implement Regularization ===============
% Once your backpropagation implementation is correct, you should now
% continue to implement the regularization with the cost and gradient.
%

fprintf('=======================================\n');
fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

% Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 3): %f ' ...
         '\n(this value should be about 0.607902)\n\n'], debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 10: Training NN ===================
% You have now implemented all the code necessary to train a neural 
% network. To train your neural network, we will now use "fmincg", which
% is a function which works similarly to "fminunc". Recall that these
% advanced optimizers are able to train our cost functions efficiently as
% long as we provide them with the gradient computations.
%

fprintf('=======================================\n');
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', numOfIter);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, finalLambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params (unrolling)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 11: Visualize Weights =================
% We can visualize our Theta1 but it usually makes no sense for us.
% Basically, Theta1 helps our program to create its own parameters to
% classify digits.

fprintf('=======================================\n');
fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 12: Implement Predict =================
% After training the neural network, we would like to use it to predict
% the labels. You will now implement the "predict" function to use the
% neural network to predict the labels of the training set. This lets
% you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('=======================================\n');
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ================= Part 13: Save Weights ===============
% Saving weights means saving the neural network we trained.
fprintf('=======================================\n');
fprintf('\nSave Weights under the name of Trained_NN.mat\n');
save('lib\Trained_NN.mat', 'Theta1', 'Theta2');

%% ================= Part 14: Check Some Random Samples ===============
% If training set accuracy and test set accuracy are two different from
% each other that means our code suffering from overfitting.
% 

fprintf('=======================================\n');
fprintf('\nCalculating training set accuracy.\n');

test_set_pred = predict(Theta1, Theta2, test_set);

fprintf('\nTesting Set Accuracy on test set: %f\n', ...
    mean(double(test_set_pred == test_set_ans)) * 100);

%% ================= Part 15: Check Some Random Samples ===============
% for i=1:100
%     displayData(test_set(i,:));
%     tic
%     what_digit = predict(Theta1, Theta2, test_set(i,:));
%     time_Val = toc;
%     fprintf('\nPredicted answer is %d. Predicted in %f seconds. Real answer is %d.\n', mod(what_digit, 10), time_Val, mod(test_set_ans(i, :), 10));
%     fprintf('\nPress enter to predict the next image. Press Ctrl+C to terminate.\n');
%     pause;
% end