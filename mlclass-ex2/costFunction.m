function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Jsum = 0;
% Gsum = 0;
% for i=1:m
%     h_tx = sigmoid(X(i,:)*theta);
%     Jsum = Jsum - y(i)*log(h_tx) - (1-y(i)) * log(1 - h_tx);
%     Gsum = Gsum + (h_tx - y(i))*X(i,:);
% end
% J = (1/m) * Jsum;
% grad = (1/m) * Gsum;

H_tx = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(H_tx) - (1-y).*(log(1-H_tx)));

H_tx_my = repmat((H_tx - y),1,size(X,2));
grad = (1/m)*sum(H_tx_my.*X);

% =============================================================

end
