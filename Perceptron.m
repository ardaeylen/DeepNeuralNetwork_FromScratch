
%training examples 
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

target = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1;];
concatenated_input = concat_input([zero;one;two;three; four; five; six; seven; eight; nine]);
flattened_input = flatten_input(concatenated_input);
weights = rands(10, 35);
alpha = 0.001;
for i=1:1000
    result = threshold_function((weights * flattened_input));
    disp(result);
    weights = weights - alpha * (result - target)* transpose(flattened_input); 
end

function [thresholded_matrix] = threshold_function(matrix)
thresholded_matrix = zeros(size(matrix));
for i=1:size(matrix,1)
    for j=1:size(matrix,2)
        if(matrix(i,j) >= 0)
            thresholded_matrix(i,j) = 1;
        end
    end
end
end


function [flattened_vector] = flattenFunction(matrix)
flattened_vector = [];    
for i = 1: size(matrix, 1)
        row = matrix(i, :);
        flattened_vector = [flattened_vector, row];
end
flattened_vector = flattened_vector.';
end
function [flattened_input] = flatten_input(concatenated_input)
flattened_input = zeros(size(concatenated_input,1) * size(concatenated_input, 2), size(concatenated_input, 3));
for i=1:size(concatenated_input,3)
flattened_input(:,i) = flattenFunction(concatenated_input(:, :, i));
end
end


%----------------

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