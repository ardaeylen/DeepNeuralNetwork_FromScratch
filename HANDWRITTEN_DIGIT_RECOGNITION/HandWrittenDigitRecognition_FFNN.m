%Hand Written digit recognition using Perceptron library.
% Shape of X_train is 256 x 1707 (1707 training examples of shape 16x16x1 -> gray level)
X_train = cell2mat(struct2cell(load("azip.mat")));

% Shape of y_train is 1 x 1707 which means the labels of classes are 1 - 10
% We have to one hot encode them to make classification across 10 classes. 
y_train = cell2mat(struct2cell(load("dzip.mat")));

% Shape of X_train is 256 x 2007 (2007 training examples of shape 16x16x1 -> gray level)
X_test = cell2mat(struct2cell(load("testzip.mat")));
y_test = cell2mat(struct2cell(load("dtest.mat")));

number_of_classes = 10;
classes = [0; 1; 2; 3; 4; 5; 6; 7; 8; 9];

y_categorical = categorical(y_train, classes);
%---------- Options-----------------
options = trainingOptions('sgdm', ... % Sthocastic Gradient Descent Learning Algorithm Used
    'MaxEpochs',100,...
    'InitialLearnRate',1e-3, ... % Initial Learning Rate = 0,001
    'Verbose',true, ...
    'Plots','training-progress');

%-----------------------------------


% Building network model.
input_layer = featureInputLayer(size(X_train,1));

hidden_layer_0 = fullyConnectedLayer(50,'WeightsInitializer','he');
hidden_layer1 = fullyConnectedLayer(50, 'WeightsInitializer', 'he');
hidden_layer2 = fullyConnectedLayer(50, 'WeightsInitializer', 'he');

output_layer = fullyConnectedLayer(number_of_classes, 'WeightsInitializer', 'he');
softmax_layer = softmaxLayer; 
classification_layer = classificationLayer('Name','output');
layers = [input_layer hidden_layer_0 hidden_layer1 hidden_layer2 output_layer softmax_layer classification_layer];
%----------------------------------------------------
%---Train the network--------------------------------
net = trainNetwork(transpose(X_train),y_categorical,layers,options);
%---------------------------------------------------

y_predicted = classify(net, transpose(X_test));
final_accuracy = sum(y_predicted == categorical(y_test)) / size(y_test,2); 























