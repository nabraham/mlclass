function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
Z2 = Theta1 * [ones(m,1) X]';
A2 = sigmoid(Z2);
Z3 = Theta2 * [ones(1,m); A2];
PRED = sigmoid(Z3);
[y i] = max(PRED);
p = i';

% =========================================================================


end
