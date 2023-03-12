%input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 30, 20, 50];
zero = [0, 1, 1, 1, 0;
        1, 0, 0, 0, 1;
        1, 0, 0, 0, 1;
        1, 0, 0, 0, 1;
        1, 0, 0, 0, 1;
        1, 0, 0, 0, 1;
        0, 1, 1, 1, 0];

one = [0, 0, 1, 0, 0;
       0, 1, 1, 0, 0;
       1, 0, 1, 0, 0;
       0, 0, 1, 0, 0;
       0, 0, 1, 0, 0;
       0, 0, 1, 0, 0;
       1, 1, 1, 1, 1];

two = [0, 1, 1, 1, 0;
       1, 0, 0, 1, 0;
       0, 0, 0, 1, 0;
       0, 0, 0, 1, 0;
       0, 0, 1, 0, 0;
       0, 1, 0, 0, 0;
       1, 1, 1, 1, 1];

three = [0, 1, 1, 1, 1;
         1, 0, 0, 0, 1;
         0, 0, 0, 0, 1;
         0, 1, 1, 1, 0;
         0, 0, 0, 0, 1;
         1, 0, 0, 0, 1;
         0, 1, 1, 1, 1];

four = [0, 0, 0, 0, 1;
        0, 0, 0, 1, 0;
        0, 0, 1, 0, 0;
        0, 1, 0, 1, 0;
        1, 1, 1, 1, 1;
        0, 0, 0, 1, 0;
        0, 0, 0, 1, 0];

five = [0, 1, 1, 1, 1;
        1, 0, 0, 0, 0;
        1, 0, 0, 0, 0;
        1, 1, 1, 1, 1;
        0, 0, 0, 0, 1;
        0, 0, 0, 0, 1;
        1, 1, 1, 1, 1];

six = [0, 1, 1, 1, 1;
       1, 0, 0, 0, 0;
       1, 0, 0, 0, 0;
       1, 1, 1, 1, 0;
       1, 0, 0, 0, 1;
       1, 0, 0, 0, 1;
       0, 1, 1, 1, 0];

seven = [1, 1, 1, 1, 1;
         0, 0, 0, 1, 0;
         0, 0, 1, 0, 0;
         0, 1, 1, 1, 0;
         0, 0, 1, 0, 0;
         0, 0, 1, 0, 0;
         0, 0, 1, 0, 0];

eight = [0, 1, 1, 1, 0;
         1, 0, 0, 0, 1;
         1, 0, 0, 0, 1;
         0, 1, 1, 1, 0;
         1, 0, 0, 0, 1;
         1, 0, 0, 0, 1;
         0, 1, 1, 1, 0];


nine = [0, 1, 1, 1, 0;
        1, 0, 0, 0, 1;
        1, 0, 0, 0, 1;
        0, 1, 1, 1, 1;
        0, 0, 0, 0, 1;
        0, 0, 0, 0, 1;
        0, 1, 1, 1, 0];
input = concat_input([zero; one; two; three; four; five; six; seven; eight; nine]);
input = flatten_input(input);
%labels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25,61, 41, 101];
labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
model = letItLearn(input, labels, [10], 10000, 0.001, 'linear','softmax');
prediction = predict_input(one, model{1}, model{2}, 'linear', 'softmax'); % predict_input(input, weights, biases, layer_activations, last_layer_activation)
% It is okey
function [flattened_vector] = flattenFunction(matrix)
flattened_vector = [];    
for i = 1: size(matrix, 1)
        row = matrix(i, :);
        flattened_vector = [flattened_vector, row];
end
flattened_vector = flattened_vector.';
end

function [concatenated_input] = concat_input(input)
k= 1;
j =7;
concatenated_input = zeros(size(input, 1)/10, size(input, 2), 10);
for i=1:size(concatenated_input,3) % For all training examples:
concatenated_input(:,:,i) =  input(k:j,:);
k = k + 7;
j = j + 7;
end
end


function [flattened_input] = flatten_input(concatenated_input)
flattened_input = zeros(size(concatenated_input,1) * size(concatenated_input, 2), size(concatenated_input, 3));
for i=1:size(concatenated_input,3)
flattened_input(:,i) = flattenFunction(concatenated_input(:, :, i));
end
end

% It is okey
function [row_sum_of_matrix] = row_sum(matrix)
    sum_matrix = zeros(size(matrix, 1),1);
    for i = 1:size(matrix,1)
        for j = 1: size(matrix, 2)
            sum_matrix(i, 1) = sum_matrix(i,1) + matrix(i, j);
        end
    end
    row_sum_of_matrix = sum_matrix;
end

function [col_sum_of_matrix] =col_sum(matrix)
    sum_matrix = zeros(1, size(matrix,2));
    for i = 1:size(matrix,2)
        for j = 1: size(matrix, 1)
            sum_matrix(1,i) = sum_matrix(1,i) + matrix(i, j);
        end
    end
    col_sum_of_matrix = sum_matrix;
end

%It is okey
function [derivative_of_g] = derivative_of_activations(z,function_name)
if(strcmp(function_name, 'sigmoid'))
    derivative_of_g = activation_function(z, function_name).*(1-activation_function(z, function_name));
end
if(strcmp(function_name, 'tanh'))
    derivative_of_g = 1 - activation_function(z, function_name).^2;
end
if(strcmp(function_name,'linear'))
    derivative_of_g = rdivide(z,z);
end
if (strcmp(function_name, 'softmax'))
    derivative_of_g = activation_function(z, function_name).*(1-activation_function(z, function_name));
end
end
% It is okey
function [activation] =  activation_function(z, function_name)
    if(strcmp(function_name, 'sigmoid'))
    activation = rdivide(1, (1+ exp(-z)));
    
    elseif(strcmp(function_name, 'tanh'))
    activation = tanh(z);
    
    elseif(strcmp(function_name, 'linear'))
    activation = z;
    
    elseif(strcmp(function_name, 'binary_step'))
    for i=1:size(z,1)
        for j = 1:size(z,2)
            if z(i,j) > 0
                z(i,j) = 1;
            else
                z(i,j) = 0;
            end
        end
    end
    elseif(strcmp(function_name, 'softmax'))
        activation = softmax_activation(z);
    end
end

function [z,activations, weights, biases] =   forward_propagation(a_prev, weights, biases,activation_function_name)
    %Forward Propagation of a single layer is executed here.
    % a_prev.shape = (n_x, m) where n_x is the number of features and m is
    % number of examples, weights.shape = (n_l, n_x) where n_l is the
    % number of neurons in the layer and n_x is the number of features that
    % is taken by the previous layer and b is just a bias for every neuron
    % in the layer

    %Forward Propagation:
    % activations = W*a_prev + biases

    z = weights * a_prev + biases;
    activations = activation_function(z, activation_function_name);
    weights = weights;
    biases = biases;
end

function [da_Prev, new_Weights, new_Biases ] = backward_propagation(da, weights, biases,a_prev,z, learning_rate, layer_activations)
% da is the derivative of the loss with respect to the current layer's activations 
% This function calculates dW ,dB (derivative of loss wirth respect to
% loss function) to implement gradient descent on weights and biases and also
%returns new weights and new biases.
%Note that this function also returns gradient of the loss function with
%respect to the previous layers activations for this function to be used in
%a multi layer manner.
%if layer_num == total_layers % If the backward propagation just begin 
 
 %   dZ = 2 * (activation - y_true);

%else
 dZ = da.* derivative_of_activations(z, layer_activations);% dZ = da * g'(Z) = (dL/ da) * (da/dZ) 

%end
    dW = rdivide( dZ * (a_prev).' , 16);
    dB = rdivide(row_sum(dZ), 16);
    new_Weights = weights - (learning_rate * dW);
    new_Biases = biases - (learning_rate * dB);
    da_Prev = (weights.') * dZ; 
    
end

% it is okey
function [weight_list, bias_list] = initialize_network(layers, input_dim)
weight_list = cell(1,size(layers, 2));
bias_list = cell(1, size(layers, 2));
first_layer_bias = [rands(layers(1), 1)];
first_layer_weight = [rands(layers(1), input_dim)];
weight_list{1, 1} = first_layer_weight; 
bias_list{1, 1} = first_layer_bias; 

if size(layers, 2) > 1 %Check whether the number of layers is greater than 1.
        for i = 2:size(layers, 2)   
            weight = [rands(layers(i), layers(i - 1))];
            bias = [zeros(layers(i), 1)];
            weight_list{i}= weight;
            bias_list{i} = bias;
            
        end
end
end

function [loss] = calculateLoss(predicted, actual, lastLayerActivationFunction)
    if strcmp(lastLayerActivationFunction,'sigmoid')
        loss = rdivide(row_sum(-(actual.* log(predicted) + (1 - actual)*log(1 - predicted))), size(actual,2));
    elseif strcmp(lastLayerActivationFunction,'linear')
        loss= rdivide(row_sum((actual - predicted).^2).^(0.5),size(actual,2));
    elseif strcmp(lastLayerActivationFunction, 'softmax')
        loss = rdivide(row_sum(col_sum(-(actual).*log(predicted))), size(actual, 2));
    end    
    loss_string = sprintf('Loss = %.5f ',(loss));
    disp(loss_string);
end

function [da] = gradientOfLossWithRespToLastLayersActivations(lastActivations, actual_values ,lossFunction)
if strcmp(lossFunction, 'binary_crossentropy')
    da = rdivide(actual_values, lastActivations) * (-1) + rdivide(1 - actual_values, 1 - lastActivations);% loss = -(y * log(a) + (1 - y) * log(1 - a)).
elseif strcmp(lossFunction, 'mean_squared_error')
    da =  -(2 /size(actual_values,2))*(actual_values - lastActivations); % loss = sum ([(a - y)^2]) / m where m is the number of training examples
elseif strcmp(lossFunction, 'categorical_crossentropy')
    da = rdivide(actual_values, lastActivations) * (-1);% loss = -sum(ylog(a)).
    %In the specific (and usual) case of Multi-Class classification 
    % the labels are one-hot, so only the positive class
    %keeps its term in the loss.
end
end


function [probabilities] = softmax_activation(z)
probabilities = zeros(size(z));
for i = 1:size(z,2)
    outputs = z(:,i);
    sum = 0;
    for j = 1: size(outputs)
    sum = sum + exp(outputs(j));
    end
    probabilities(:,i) = rdivide(exp(outputs),sum); 
end   
    
end



function [model] = letItLearn(input,labels,layers, epochs, learning_rate, layer_activations, last_layer_activation)

[weights, biases] = initialize_network(layers,size(input,1));

activations = cell(1, size(layers,2)+1);
activations{1} = input; %First activation is the input
weighted_inputs = cell(1, size(layers,2));% Z values are stored for backward propagation

for i = 1:epochs
    for layer = 1: size(layers, 2)
        if layer ~= size(layers,2)
        [weighted_inputs{layer},activations{layer + 1}, weights{layer}, biases{layer}] = forward_propagation(activations{layer}, weights{layer}, biases{layer}, layer_activations);
        else
        [weighted_inputs{layer},activations{layer + 1}, weights{layer}, biases{layer}] = forward_propagation(activations{layer}, weights{layer}, biases{layer}, last_layer_activation);
        end    
    end
    [loss] = calculateLoss(activations{layer + 1}, labels, last_layer_activation);
    [da] = gradientOfLossWithRespToLastLayersActivations(activations{layer +1 }, labels,'categorical_crossentropy');
    %disp(da);
    for layer = size(layers, 2):-1:1
        [da, weights{layer}, biases{layer}] = backward_propagation(da, weights{layer}, biases{layer},activations{layer},weighted_inputs{layer}, learning_rate, layer_activations);
        %disp(da);                                                 
    end

end
model = {weights, biases};
end

function [prediction] = predict_input(input, weights, biases, layer_activations, last_layer_activation)
activations = {};
activations{1} = flatten_input(input);
for layer = 1: size(weights, 2)
        if layer ~= size(weights,2)
        [weighted_inputs{layer},activations{layer + 1}, weights{layer}, biases{layer}] = forward_propagation(activations{layer}, weights{layer}, biases{layer}, layer_activations);
        else
        [weighted_inputs{layer},activations{layer + 1}, weights{layer}, biases{layer}] = forward_propagation(activations{layer}, weights{layer}, biases{layer}, last_layer_activation);
        end    
end
prediction = activations{size(weights,2) + 1};
end