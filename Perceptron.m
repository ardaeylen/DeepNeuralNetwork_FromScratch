input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 30, 20, 50];
labels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25,61, 41, 101];
model = letItLearn(input, labels, [1,3,1], 100, 0.001, 'sigmoid','linear');

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
%It is okey
function [derivative_of_g] = derivative_of_activations(z,function_name)
if(strcmp(function_name, 'sigmoid'))
    derivative_of_g = activation_function(z, 'sigmoid').*(1-activation_function(z, 'sigmoid'));
end
if(strcmp(function_name, 'tanh'))
    derivative_of_g = 1 - activation_function(z, 'tanh').^2;
end
if(strcmp(function_name,'linear'))
    derivative_of_g = rdivide(z,z);
end
end
% It is okey
function [activation] =  activation_function(z, function_name)
    if(strcmp(function_name, 'sigmoid'))
    activation = rdivide(1, (1+ exp(-z)));
    end
    if(strcmp(function_name, 'tanh'))
    activation = tanh(z);
    end
    if(strcmp(function_name, 'linear'))
    activation = z;
    end
    if(strcmp(function_name, 'binary_step'))
    for i=1:size(z,1)
        for j = 1:size(z,2)
            if z(i,j) > 0
                z(i,j) = 1;
            else
                z(i,j) = 0;
            end
        end
    end
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

function [da_Prev, new_Weights, new_Biases ] = backward_propagation(da, weights, biases,a_prev,z, layer_num, total_layers, activation, y_true, learning_rate)
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
 dZ = da.* derivative_of_activations(z, 'linear');% dZ = da * g'(Z) = (dL/ da) * (da/dZ) 

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
    loss = r_divide(row_sum(-(actual * log(predicted) + (1 - actual)*log(1 - predicted))), size(actual,2));
elseif strcmp(lastLayerActivationFunction,'linear')
    loss= rdivide(row_sum((actual - predicted).^2).^(0.5),size(actual,2));
end    
    loss_string = sprintf('Loss = %.5f ',(loss));
    disp(loss_string);
end

function [da] = gradientOfLossWithRespToLastLayersActivations(lastActivations, actual_values ,lossFunction)
if strcmp(lossFunction, 'binary_crossentropy')
    da = rdivide(actual_values, lastActivations) * (-1) + rdivide(1 - actual_values, 1 - lastActivations);% loss = -(y * log(a) + (1 - y) * log(1 - a)).
elseif strcmp(lossFunction, 'mean_squared_error')
    da =  -(2 /size(actual_values,2))*(actual_values - lastActivations); % loss = sum ([(a - y)^2]) / m where m isthe number of training examples
    
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
    [loss] = calculateLoss(activations{layer + 1}, labels, 'linear');
    [da] = gradientOfLossWithRespToLastLayersActivations(activations{layer +1 }, labels,'mean_squared_error');
    for layer = size(layers, 2):1
        [da, weights{layer}, biases{layer}] = backward_propagation(da, weights{layer}, biases{layer},activations{layer},weighted_inputs{layer}, layer, size(layers,2), activations{layer + 1}, labels, learning_rate);
        
    end

end
model = {weights, biases};
end
