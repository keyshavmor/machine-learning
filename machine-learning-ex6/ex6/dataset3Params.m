function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% =========================================================================

suggest = [0.01,0.03,0.1,0.3,1,3,10,30];

x1 = X(1,:);
x2 = X(2,:);
x1 = x1(:)
x2 = x2(:)

meanError = zeros(length(suggest),length(suggest));

for i = 1:length(suggest)
    C = suggest(i); 
    for j = 1:length(suggest)
        sigma = suggest(j); 
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        prediction = svmPredict(model,Xval);
        meanError(i,j) = mean(double(prediction ~= yval));
    end
end

[minVal,idx] = min(meanError,[],2);

[minVal,idx2] = min(minVal,[],1);

C = suggest(idx2);

minError = zeros(length(idx),2);

minError(:,1) = 1:length(idx);
minError(:,2) = idx; 

for i = 1:length(idx)
    if idx2 == minError(i,1),
       sigma = suggest(minError(i,2));
    end
end

end
