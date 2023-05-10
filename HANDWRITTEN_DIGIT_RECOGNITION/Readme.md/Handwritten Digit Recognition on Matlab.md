In this assignment, we were wanted to train a multi-layer perceptron that recognizes simple handwritten digits with a flexibility of deciding architecture and hyperparameters by ourselves. 

## Training and Test Data

### Training Data

- Training data consists of 1707 samples of handwritten digit images with corresponding categorical labels.
- Shape of each of the input training example is (16x16x1) and it was flattened.
- Output training examples are categorical (0-9) numbers represents the corresponding input training example.

### Test Data
- Test data consists of 2007 examples that are identical in shape with training data.

## Experiments Results

### Initial Network Configuraitons
- Learning Rate = 0.001.
- All layers' weights are initialized with 'He' initialization method.
- Network consists of 3 hidden layers each have 50 hidden untis (neurons) within them.

After training first feed-forward neural network with Gradient Descent with Momentum learning algorithm (To converge global minimum fast) with constant learning rate (0.001) test accuracy is observed as 87.74% and for this reason epoch number is incremented from 100 to 200.

![[TrainingPerformance.png]]
		**Training performance of 100 epochs of the first feed forward neural network model.**
		
As expected, increasing the number of epochs did not avoid saturation of accuracy. Accuracy increased to 88.39% so to avoid saturation of accuracy, neuron or hidden unit size of the hidden layers are increased but it only led to decreasing accuracy.

![[TrainingPerformance200epochs.png]]
	**Training performance of 200 epochs of the same model.**

After changing the optimization algorithm from Gradient Descent with Momentum to Adaptive Moment Estimation to converge global minima even faster and less oscillation, test accuracy did not change at all. Even increasing network depth did not helped.