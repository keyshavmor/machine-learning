function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
squaredTheta = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = X*theta; %regular Linear regression

hypothesis = sigmoid(hypothesis); % logistic regression

dTheta=(X'*(hypothesis-y));

grad = (dTheta/m);

errorVal = (log(hypothesis).*(-y)) - (log(1 - hypothesis).*(1-y));

J = (sum(errorVal)/m);

n = length(theta);

for i = 2:n

      squaredTheta = squaredTheta + theta(i)^2; 
      grad(i) = grad(i) + (lambda*theta(i)/m);
end

regularisedTheta = ((lambda*squaredTheta)/(2*m));

J = J + regularisedTheta

grad


% =============================================================

end
