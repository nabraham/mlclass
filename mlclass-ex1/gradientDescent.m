function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	 %

     %loop style
% 	 for j=1:size(X,2)
% 	    gsum = 0;
% 	    for i=1:m
% 			 gsum = gsum + (X(i,:)*theta - y(i)) * X(i,j);
% 		end
% 		theta(j) = theta(j) - (1/m) * alpha * gsum;
%      end
    
     %vectorized
%      H = X*theta - y;
%      HR = repmat(H,1,size(X,2));
%      SHRX = sum(HR.*X);
%      theta = theta - (1/m) * alpha*SHRX';
       theta = theta - (1/m) * alpha * sum(repmat(X*theta-y,1,n).*X)';
     

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
