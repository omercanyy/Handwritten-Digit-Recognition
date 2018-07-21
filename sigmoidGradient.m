function g = sigmoidGradient(z)
% SIGMOIDGRADIENT returns the gradient of the sigmoid function
% evaluated at z. This should work regardless if z is a matrix or a vector.

% ====================== YOUR CODE HERE ======================


g = sigmoid(z).*(1-sigmoid(z));


% =============================================================




end
