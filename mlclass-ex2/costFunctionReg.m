function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
thetaLamdaVec = theta;
thetaLamdaVec(1) = 0;
thetaLamdaVec = thetaLamdaVec * lambda;
% You need to return the following variables correctly 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Jsum = 0;
% Gsum = 0;
% 
% for i=1:m
%     h_tx = sigmoid(X(i,:)*theta);
%     Jsum = Jsum - y(i)*log(h_tx) - (1-y(i)) * log(1 - h_tx);
%     Gsum = Gsum + (h_tx - y(i))*X(i,:);
% end
% J = (1/m) * Jsum + (lambda/(2*m))*sum(theta(2:end).^2);
% grad = (1/m) * (Gsum + thetaLamdaVec');

H_tx = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(H_tx) - (1-y).*(log(1-H_tx))) + (lambda/(2*m))*sum(theta(2:end).^2);
 
H_tx_my = repmat((H_tx - y),1,size(X,2));
grad = (1/m)*(sum(H_tx_my.*X) + thetaLamdaVec');

% =============================================================

end
