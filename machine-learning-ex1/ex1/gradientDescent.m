function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

tmp = zeros(2,1); 
       
for iter = 1:num_iters
     
 % for i = 1:2
 %   if i == 1 ,
 %      dTheta=sum(((X*theta)-y));
 %      dThetamean=(dTheta/m);
 %      tmp(i)=theta(i)-(alpha*dThetamean);
 %   else 
 %      dTheta=sum(((X*theta)-y).*X(:,2));
 %      dThetamean=(dTheta/m);
 %      tmp(i)=theta(i)-(alpha*dThetamean);
 %   end; 
 % end  
 %  theta(1)= tmp(1);
 %  theta(2)= tmp(2);

    dTheta=X'*((X*theta)-y);
    dThetamean=(dTheta/m);
    theta=theta-(alpha*dThetamean);
     
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %






    % ============================================================

    % Save the cost J in every iteration    
  J_history(iter) = computeCost(X, y, theta);


end

end
