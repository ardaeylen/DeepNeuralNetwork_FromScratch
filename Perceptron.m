layers = [1];
[weights, biases] = initialize_network(layers,1);
in = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 30, 20, 50];
y_true = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25,61, 41, 101];
epochs = 10000;
activations = cell(1, size(layers,2)+1);
activations{1} = in;
weighted_inputs = cell(1, size(layers,2));

learning_rate = 0.0001;
for i = 1:1000
    for layer = 1: size(layers, 2)
        [weighted_inputs{layer},activations{layer + 1}, weights{layer}, biases{layer}] = forward_propagation(activations{layer}, weights{layer}, biases{layer});
       
    end
    for layer = size(layers, 2):1
        [a_prev, weights{layer}, biases{layer}] = backward_propagation(in, weights{layer}, biases{layer},activations{layer},weighted_inputs{layer}, layer, size(layers,2), activations{layer + 1}, y_true, learning_rate);
        
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
%It is okey
function [derivative_of_g] = derivative_of_activations(z,function_name)

    derivative_of_g = activation_function(z).*(1-activation_function(z));
end
% It is okey
function [activation] =  activation_function(z)
    activation = rdivide(1, (1+ exp(-z)));
end

function [z,activations, weights, biases] =   forward_propagation(a_prev, weights, biases)
    %Forward Propagation of a single layer is executed here.
    % a_prev.shape = (n_x, m) where n_x is the number of features and m is
    % number of examples, weights.shape = (n_l, n_x) where n_l is the
    % number of neurons in the layer and n_x is the number of features that
    % is taken by the previous layer and b is just a bias for every neuron
    % in the layer

    %Forward Propagation:
    % activations = W*a_prev + biases

    z = weights * a_prev + biases;
    activations = z; %activation_function(z);
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
if layer_num == total_layers
 
    dZ =2 * (activation - y_true) / 16; % Because linear function is used
    %disp('activations:')
    %disp(activation)
    %disp("Actual values")
    %disp(y_true)
else
    dZ = da.* derivative_of_activations(z, 'sigmoid');
end
    dW = dZ * (a_prev).';
    %disp(dW);
    %disp(size(weights));
    dB = row_sum(dZ);
    new_Weights = weights - (learning_rate * dW);
    new_Biases = biases - (learning_rate * dB);
    da_Prev = (weights.') * dZ;
    disp((new_Weights));
end

% it is okey
function [weight_list, bias_list] = initialize_network(layers, input_dim)
weight_list = cell(1,size(layers, 2));
bias_list = cell(1, size(layers, 2));
first_layer_bias = [rands(layers(1), 1)];
first_layer_weight = [rands(layers(1), input_dim)];
weight_list{1, 1} = first_layer_weight; 
bias_list{1, 1} = first_layer_bias; 
if size(layers, 2) > 1
        for i = 2:size(layers, 2)   
            weight = [rands(layers(i), layers(i - 1))];
            bias = [zeros(layers(i), 1)];
            weight_list{i}= weight;
            bias_list{i} = bias;
            
        end
    end
end



