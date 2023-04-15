%Hand Written digit recognition using Perceptron library.
X_train = cell2mat(struct2cell(load("azip.mat")));
y_train = cell2mat(struct2cell(load("dzip.mat")));


X_test = cell2mat(struct2cell(load("testzip.mat")));
y_test = cell2mat(struct2cell(load("dtest.mat")));

number_of_classes = 10;

%One - Hot Encoding processing
%Since there are 10 classes (0-9) we must pass this information to the 
y_train = one_hot_encode(y_train, number_of_classes);
y_test_encoded = one_hot_encode(y_test, number_of_classes);


% Building network model.

singleLayerPerceptron = perceptron;

singleLayerPerceptron = train(singleLayerPerceptron, X_train, y_train);

%view(singleLayerPerceptron);
% One-Hot encoding function------------------------------------------
function [one_hot_encoded] = one_hot_encode(labels, number_of_classes)
    one_hot_encoded = zeros(number_of_classes,size(labels,2));
    for i = 1:size(labels, 2)
        one_hot_encoded(labels(1,i) + 1,i) = 1;
    end
end
%-----------------------------------------------------------------------


