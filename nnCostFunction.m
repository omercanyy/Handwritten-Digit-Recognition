function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% m ýs the number of training examples.
m = size(X, 1);

% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
% Part 3: Implement regularization with the cost function and gradients.
%
% ====================== YOUR CODE HERE ======================

a1 = [ones(size(X, 1),1) X];
a2 = [ones(size(a1, 1), 1) sigmoid(a1*Theta1')];
a3 = sigmoid(a2*Theta2.');

hx = a3;

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

J = (1/m)*sum(sum((-y_matrix.*log(hx)-(1-y_matrix).*log(1-hx))))...
    +(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

d3 = a3 - y_matrix;
d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(a1*Theta1.');

Delta1 = d2'*a1;
Delta2 = d3'*a2;

Theta1_grad = Delta1/m...
    + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]...
    ;
Theta2_grad = Delta2/m...
    + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]...
    ;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
