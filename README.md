# Computational-Intelligence

My solutions to Computational Intelligence Projects, such as Fuzzy Logic, Hopfield Network for Bit String Recognition ,Fuzzy C-Means Clustering, and Neural Network 
, Fall 2023, Dr.Mozaieni.

## <img width="40" height="40" src="https://img.icons8.com/?size=100&id=kOPTH4LnJoIU&format=png&color=000000" alt="homework"/> Projects
### <img width="40" height="40" src="https://img.icons8.com/?size=100&id=104091&format=png&color=000000" /> P1
Array and Matrix Operations
This repository contains solutions to various array and matrix manipulation problems using Python and NumPy.

Project Questions and Solutions
1. Element-wise Comparison of Arrays
•  Function to compare two arrays element-wise (greater, greater or equal, less, less or equal) using NumPy.
2. Element-wise and Matrix Multiplication
•  Function to perform element-wise or matrix multiplication based on a specified method using NumPy.
3. Array Addition
•  Function to add a vector to a matrix either horizontally or vertically based on a specified method.
4. Matrix Normalization
•  Create a 4x4 random matrix with values between 1 and 10, then normalize the values to be between 0 and 1.
5. CSV Data Analysis
•  Read data from a CSV file and perform various calculations:

•  Calculate daily returns.

•  Compute mean and standard deviation of daily returns.

•  Plot daily closing prices and returns.

•  Identify days with highest and lowest returns.

•  Find historical highest and lowest stock prices.

1. Forward Feed in Neural Networks
•  Implement forward feed operation for 1000 samples with 500 features each using both for loops and vectorization. Compare the performance of both methods.
2. Threshold-based Array Modification
•  Function to modify array elements based on a threshold value without using loops, utilizing NumPy.
3. Matrix Data Structure Class
•  Define a class for matrix data structure with methods to:

•  Check equality with another matrix.

•  Compare elements with another matrix.

•  Check if one matrix is a subset of another.


- Answers: [Link to DL1](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_1_SaraYounesi)

### <img width="40" height="40" src="https://img.icons8.com/?size=100&id=104091&format=png&color=000000" />  P2
- 
Neural Network and Machine Learning Exercises
This repository contains solutions to various neural network and machine learning problems using Python and NumPy.

Project Questions and Solutions
1. Adaline Neuron for NAND Gate
•  Model the NAND gate using an Adaline neuron with a learning rate of 0.1, random initial weights, and biases. Train the model for 4 epochs and document all training steps.
2. Activation Functions and MLP Training
•  a. Difference between linear and non-linear activation functions.

•  b. Impact of different initial weights and biases on MLP training:

•  Random biases and zero weights.

•  Zero biases and random weights.

•  c. Generalization capabilities of various neural networks.

•  d. Advantages and disadvantages of using the given weight update formula in MLP training.

1. MLP with Different Activation Functions
•  Create an MLP with one hidden layer and three neurons for four datasets. Train the model with different activation functions up to 500 epochs and analyze the impact on each dataset.
2. Notebook Analysis
•  Complete and analyze the provided notebook (2-4HW), run the cells, and interpret the generated code and plots.
3. Multi-layer Perceptron for XNOR Function
•  Train a multi-layer perceptron using NumPy to learn the XNOR function.
4. MLP on MNIST Dataset
•  Design an MLP for the MNIST dataset to achieve at least 95% accuracy. Explain the choice of layers and neurons, and plot the loss and accuracy using Matplotlib.

- Answers: [Link to DL2](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_2_SaraYounesi)

### <img width="40" height="40" src="https://img.icons8.com/?size=100&id=104091&format=png&color=000000" /> P3
Neural Network Exercises and Solutions
This repository contains solutions to various neural network problems using different models such as Kohonen, Hopfield, and MLP.

Project Questions and Solutions
1. Kohonen Network for Clustering
•  Train a Kohonen network to classify four input neurons into two categories with an initial learning rate of 0.5. Document the training steps for two points and outline the process for the remaining steps.
2. Hopfield Network Local Minima
•  Determine if the list ([1,1,-1,-1], [-1,-1,1,1], [-1,-1,-1,-1], [1,1,1,1]) can be stored as local minima in a Hopfield network. If not, explain why. If yes, calculate the network weights.
3. MLP for Function Approximation
•  Implement and train an MLP to approximate the function (y = 2^x). Compare the neural network output with the actual function output in the range ([-3, 3]).
4. Hopfield Network for Bit String Recognition
•  Design a Hopfield network to converge from the bit string 010000 to 111100. Provide the weight matrix and computation table.
5. Traveling Salesman Problem (TSP)
•  Discuss how the TSP can be solved using neural networks (Hopfield, SOM, MLP). Provide the algorithm, network structure, and other relevant details if solvable. If not, explain why.
- Answers: [Link to DL3](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_3_SaraYounesi)

### <img width="40" height="40" src="https://img.icons8.com/?size=100&id=104091&format=png&color=000000" /> P4
Fuzzy Logic and Neural Network Exercises
This repository contains solutions to various problems involving fuzzy logic and neural networks.

Project Questions and Solutions
1. De Morgan's Laws in Fuzzy Sets
•  Verify De Morgan's laws for the given fuzzy sets (A), (B), and (C) with specified membership functions and operators (NOT, AND, OR).
2. Fuzzy Rule for Volume and Pressure
•  Analyze the fuzzy rule: "If volume is very low, then pressure is very high." Given membership functions for volume and pressure, determine the membership degrees for high pressure when volume is not fairly low.
3. Fuzzy Controller for Autonomous Vehicle
•  Design a fuzzy controller to determine the pressure on the gas pedal based on the distance from the car in front and road slipperiness. The inputs have three states (low, medium, high), and the output has five states (very low, low, medium, high, very high).

•  a. Describe the steps of a fuzzy controller.

•  b. Plot the membership functions for the two input variables.

•  c. Calculate the gas pedal pressure for given input values (distance = 0.65, slipperiness = 0.5).

1. Defuzzification Methods
•  Discuss the advantages and disadvantages of different defuzzification methods.
- Answers: [Link to DL4](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_4_SaraYounesi)

### <img width="40" height="40" src="https://img.icons8.com/?size=100&id=104091&format=png&color=000000" /> P5
Fuzzy Logic and Neural Network Exercises
This repository contains solutions to various problems involving fuzzy logic and neural networks.

Project Questions and Solutions

----------------------------
Fuzzy Control System for Pendulum Problem
•  a. Implement a graphical environment for the pendulum problem and display the movement window.

•  b. Design a fuzzy control system using the fuzzy-scikit library to solve the pendulum problem. Define the input variables (cos(angle), sin(angle), angular velocity) and the output variable (torque). Create rules to control the pendulum and implement them in the library. Ensure the system reaches the goal within 500 steps.

•  c. Write a report including:

•  Definition and range of linguistic variables.

•  Rule definitions.

•  Reward plots and their analysis.

----------------------------
Fuzzy C-Means Clustering
•  a. Research the Fuzzy C-Means (FCM) algorithm and explain its working and differences from the classic K-Means algorithm.

•  b. Use the skfuzzy library to apply FCM on provided datasets. Normalize or standardize the features, cluster the data for different values of (c) (2 to 10), and visualize the clusters. Determine the best number of clusters using the Fuzzy Partition Coefficient (FPC) and explain the selection process.

----------------------------
Fuzzy Logic for Linguistic Variables
•  Given the membership functions for weight and age, evaluate the truth of the following statements:

•  a. The second person is relatively fatter and younger than the first person.

•  b. If the first person is very thin, then the second person is relatively young.

- Answers: [Link to DL5](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_5_SaraYounesi)

### <img width="40" height="40" src="https://img.icons8.com/?size=100&id=104091&format=png&color=000000" /> P6
Fuzzy Logic and Neural Network Solutions
This repository contains solutions to various problems involving fuzzy logic and neural networks, implemented in Python.

Project Questions and Solutions

----------------------------
Fuzzy Control System for Pendulum Problem
•  Implement a graphical environment for the pendulum problem.

•  Design a fuzzy control system using the fuzzy-scikit library to control the pendulum's movement.

•  Define input variables (cos(angle), sin(angle), angular velocity) and output variable (torque).

•  Create and implement rules to control the pendulum, ensuring it reaches the goal within 500 steps.

•  Report includes variable definitions, rule definitions, reward plots, and analysis.

----------------------------
Fuzzy C-Means Clustering
•  Research and explain the Fuzzy C-Means (FCM) algorithm and its differences from K-Means.

•  Use the skfuzzy library to apply FCM on provided datasets.

•  Normalize or standardize features, cluster data for different values of (c) (2 to 10), and visualize clusters.

•  Determine the best number of clusters using the Fuzzy Partition Coefficient (FPC) and explain the selection process.

----------------------------
Fuzzy Logic for Linguistic Variables
•  Evaluate the truth of statements given membership functions for weight and age.

•  Analyze statements about relative fatness and youthfulness based on provided membership functions.

----------------------------
Genetic Algorithm for Root Finding
•  Implement a genetic algorithm to find the roots of a function.

•  Define chromosome representation, fitness function, selection, crossover, and mutation processes.

•  Run the algorithm for a specified number of generations and analyze the results.

- Answers: [Link to DL6](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_6_SaraYounesi)




