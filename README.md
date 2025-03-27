# Feed Forward Network (FFN) on MNIST

## Overview
This project involves implementing a Feed Forward Network (FFN) using Scikit-learn and exploring various neural network design choices with Keras for digit classification on the MNIST dataset.

## Dataset
The MNIST dataset consists of 70,000 images of handwritten digits (0-9), each of size 28x28 pixels. It can be accessed from:
- [OpenML MNIST dataset](https://www.openml.org/d/554)
- [GeeksforGeeks MNIST guide](https://www.geeksforgeeks.org/mnist-dataset/)
- [GTDLBench MNIST dataset](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)

## Part 1: Implementing FFN using Scikit-learn

### Steps:
1. **Download the dataset** using `fetch_openml("mnist_784")`.
2. **Explore the dataset**:
   - Print the shape of input data `(70,000, 784)` and target data `(70,000,)`.
   - Display the first 10 images using `matplotlib`.
3. **Preprocess the dataset**:
   - Reshape the dataset into `(70,000, 28, 28)` temporarily.
   - Define the feature matrix `X` (70,000, 784) and target vector `y`.
   - Scale the dataset (normalize or standardize the features).
4. **Split the dataset** into training (80%) and testing (20%) using `train_test_split()`.
5. **Train an FFN model** using `MLPClassifier`:
   - One hidden layer with 64 neurons.
   - Set `max_iter=10`.
6. **Evaluate the model**:
   - Compute accuracy using `accuracy_score()`.
   - Compute precision, recall, and F1-score using `precision_recall_fscore_support()`.
7. **Experiment with different train-test splits** (60-40, 75-25, 80-20, 90-10) and visualize results.
8. **Tune model hyperparameters**:
   - Increase `max_iter` to 20, 50, 100, 150, and 200 and analyze accuracy variations.

## Part 2: Exploring Neural Network Design Choices using Keras

### Experiments:
1. **Varying number of nodes in a single hidden layer**:
   - Train models with 4, 32, 64, 128, 512, and 2056 nodes for 10 epochs.
   - Record training/testing accuracy, number of parameters, and training time.
2. **Varying number of hidden layers**:
   - Train networks with 5 hidden layers (64 nodes each) for 10 epochs.
   - Modify layers to 4, 6, 8, 16.
   - Run models for 30 epochs and compare results.
3. **Layer-node combinations**:
   - Experiment with architectures using different neuron distributions (e.g., 256, 128, 64, 32).
   - Determine the best-performing configuration.
4. **Effect of input size**:
   - Train a model with 4 hidden layers (256 nodes each) using ReLU activation for 10 epochs.
   - Analyze accuracy changes.
5. **Varying dataset splits**:
   - Experiment with different training and testing sizes.
   - Compare accuracy trends.
6. **Effect of activation functions**:
   - Train a network with 4 hidden layers (64 nodes each) using different activation functions:
     - Sigmoid, Tanh, ReLU.
   - Run models for 10 and 30 epochs.
   - Compare training and testing accuracy.
7. **Combining different activation functions**:
   - Use different activations in a 3-layer architecture (e.g., Sigmoid-ReLU-Tanh).
   - Identify the best activation combination for a 32-node architecture.

## Dependencies
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Keras
- TensorFlow

## Execution
Run the scripts in sequence for both parts to experiment with different configurations and compare results.

