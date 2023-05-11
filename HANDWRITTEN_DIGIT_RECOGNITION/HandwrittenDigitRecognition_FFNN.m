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
options = trainingOptions('adam', ... % Adaptive Moment Estimation Learning Algorithm Used
    'MaxEpochs',128,...
    'InitialLearnRate',1e-3, ... % Initial Learning Rate = 0,001
    'Verbose',true, ...
    'Plots','training-progress');

%-------------------------------------


% Building network model.
input_layer = featureInputLayer(size(X_train,1));

hidden_layer_1 = fullyConnectedLayer(50,'WeightsInitializer','he');
hidden_layer2 = fullyConnectedLayer(50, 'WeightsInitializer', 'he');

hidden_layer_3 = fullyConnectedLayer(50, 'WeightsInitializer', 'he');


output_layer = fullyConnectedLayer(number_of_classes, 'WeightsInitializer', 'he');
softmax_layer = softmaxLayer; 
classification_layer = classificationLayer('Name','output');
layers = [input_layer hidden_layer_1 hidden_layer2 hidden_layer_3 output_layer softmax_layer classification_layer];
%----------------------------------------------------
%---Train the network--------------------------------
net = trainNetwork(transpose(X_train),y_categorical,layers,options);
%---------------------------------------------------

y_predicted = classify(net, transpose(X_test));
final_accuracy = sum(y_predicted == categorical(transpose(y_test)),'all') / size(y_test,2); 
%---------------------------------------------------

conf_matrix = plotconfusion(categorical(transpose(y_test)), y_predicted);
conf_matrix = confusionmat(categorical(transpose(y_test)), y_predicted);

scores = statsOfMeasure(conf_matrix);
disp(scores);


function [stats] = statsOfMeasure(confusion)

tp = [];
fp = [];
fn = [];
tn = [];
len = size(confusion, 1);
for k = 1:len                  %  predict
    % True positives           % | x o o |
    tp_value = confusion(k,k); % | o o o | true
    tp = [tp, tp_value];       % | o o o |
                                               %  predict
    % False positives                          % | o o o |
    fp_value = sum(confusion(:,k)) - tp_value; % | x o o | true
    fp = [fp, fp_value];                       % | x o o |
                                               %  predict
    % False negatives                          % | o x x |
    fn_value = sum(confusion(k,:)) - tp_value; % | o o o | true
    fn = [fn, fn_value];                       % | o o o |
                                                                       %  predict
    % True negatives (all the rest)                                    % | o o o |
    tn_value = sum(sum(confusion)) - (tp_value + fp_value + fn_value); % | o x x | true
    tn = [tn, tn_value];                                               % | o x x |
end
% Statistics of interest for confusion matrix
prec = tp ./ (tp + fp); % precision
sens = tp ./ (tp + fn); % sensitivity, recall
spec = tn ./ (tn + fp); % specificity
acc = sum(tp) ./ sum(sum(confusion));
f1 = (2 .* prec .* sens) ./ (prec + sens);
% For micro-average
microprec = sum(tp) ./ (sum(tp) + sum(fp)); % precision
microsens = sum(tp) ./ (sum(tp) + sum(fn)); % sensitivity, recall
microspec = sum(tn) ./ (sum(tn) + sum(fp)); % specificity
microacc = acc;
microf1 = (2 .* microprec .* microsens) ./ (microprec + microsens);
% Names of the rows
name = ["true_positive"; "false_positive"; "false_negative"; "true_negative"; ...
    "precision"; "sensitivity"; "specificity"; "accuracy"; "F-measure"];
% Names of the columns
varNames = ["name"; "classes"; "macroAVG"; "microAVG"];
% Values of the columns for each class
values = [tp; fp; fn; tn; prec; sens; spec; repmat(acc, 1, len); f1];
% Macro-average
macroAVG = mean(values, 2);
% Micro-average
microAVG = [macroAVG(1:4); microprec; microsens; microspec; microacc; microf1];
% OUTPUT: final table
stats = table(name, values, macroAVG, microAVG, 'VariableNames',varNames);

end

