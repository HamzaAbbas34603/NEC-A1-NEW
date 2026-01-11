# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import ParameterGrid
import random

# Load the dataset
turbine_data = pd.read_csv('Preprocessed/Preprocessed_A1_turbine.csv', sep=',', header=0)

# Display basic info for sanity checks
print("Turbine Data Info:")
turbine_data.info()

# Extract features (X) and target (y) for the dataset
X_turbine = turbine_data.drop('power', axis=1).values
y_turbine = turbine_data['power'].values

# Define the neural network class
class MyNeuralNetwork:
    def __init__(self, layers, learning_rate, momentum, activation, validation_percentage=0.2):
        self.L = len(layers)
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation = activation
        self.validation_percentage = validation_percentage
        
        # Initialize weights, biases, and gradients
        self.w = [np.random.randn(layers[i], layers[i-1]) for i in range(1, self.L)]
        self.theta = [np.zeros(l) for l in layers]
        self.d_w_prev = [np.zeros_like(w) for w in self.w]
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]
        self.xi = [np.zeros(l) for l in layers]
        self.delta = [np.zeros(l) for l in layers]
        self.loss_epochs = []

    def activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'linear':
            return x
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Invalid activation function")

    def activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'linear':
            return 1
        elif self.activation == 'tanh':
            return 1 - x**2
        else:
            raise ValueError("Invalid activation function")

    def forward_pass(self, x):
        self.xi[0] = x
        for i in range(1, self.L):
            self.xi[i] = self.activation_function(np.dot(self.w[i-1], self.xi[i-1]) - self.theta[i])

    def backward_pass(self, y):
        self.delta[-1] = (self.xi[-1] - y) * self.activation_derivative(self.xi[-1])
        for i in range(self.L - 2, 0, -1):
            self.delta[i] = np.dot(self.w[i].T, self.delta[i+1]) * self.activation_derivative(self.xi[i])

    def update_weights(self):
        for i in range(1, self.L):
            d_w = self.learning_rate * np.outer(self.delta[i], self.xi[i-1]) + self.momentum * self.d_w_prev[i-1]
            d_theta = self.learning_rate * self.delta[i] + self.momentum * self.d_theta_prev[i]
            self.w[i-1] -= d_w
            self.theta[i] -= d_theta
            self.d_w_prev[i-1] = d_w
            self.d_theta_prev[i] = d_theta

    def fit(self, X, y, epochs, batch_size=32, learning_rate_decay=0.1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_percentage, random_state=42)
        for epoch in range(epochs):
            for _ in range(0, len(X_train), batch_size):
                batch_indices = np.random.choice(len(X_train), min(batch_size, len(X_train)), replace=False)
                for i in batch_indices:
                    self.forward_pass(X_train[i])
                    self.backward_pass(y_train[i])
                    self.update_weights()
            # Log training and validation error
            train_error = mean_squared_error(y_train, self.predict(X_train))
            val_error = mean_squared_error(y_val, self.predict(X_val))
            print(f"Epoch {epoch+1}/{epochs} - Train Error: {train_error:.4f}, Val Error: {val_error:.4f}")
            self.loss_epochs.append([train_error, val_error])
            self.learning_rate *= (1.0 / (1.0 + learning_rate_decay * epoch))
        return np.array(self.loss_epochs)

    def predict(self, X):
        predictions = []
        for x in X:
            self.forward_pass(x)
            predictions.append(self.xi[-1][0])
        return np.array(predictions)

# ================================
# 1. Grid Search for Hyperparameter Tuning
# ================================
param_grid = {
    "layers": [[X_turbine.shape[1], 8, 8, 1], [X_turbine.shape[1], 16, 16, 1], [X_turbine.shape[1], 32, 32, 1]],
    "learning_rate": [0.001, 0.01, 0.1],
    "momentum": [0.8, 0.9, 0.99],
    "activation": ["sigmoid", "relu", "tanh"],
}

best_model = None
best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    nn = MyNeuralNetwork(
        layers=params["layers"],
        learning_rate=params["learning_rate"],
        momentum=params["momentum"],
        activation=params["activation"]
    )
    loss = nn.fit(X_turbine, y_turbine, epochs=30)  # Use fewer epochs for faster tuning
    val_error = loss[-1][1]
    if val_error < best_score:
        best_model = nn
        best_score = val_error
        best_params = params

print("Best Parameters from Grid Search:", best_params)

# ================================
# 2. Random Search for Hyperparameter Tuning
# ================================
num_samples = 20
best_model = None
best_score = float('inf')
best_params = None

for _ in range(num_samples):
    params = {
        "layers": [X_turbine.shape[1], random.choice([8, 16, 32]), random.choice([8, 16, 32]), 1],
        "learning_rate": random.uniform(0.0001, 0.1),
        "momentum": random.uniform(0.8, 0.99),
        "activation": random.choice(["sigmoid", "relu", "tanh"]),
    }
    nn = MyNeuralNetwork(
        layers=params["layers"],
        learning_rate=params["learning_rate"],
        momentum=params["momentum"],
        activation=params["activation"]
    )
    loss = nn.fit(X_turbine, y_turbine, epochs=30)
    val_error = loss[-1][1]
    if val_error < best_score:
        best_model = nn
        best_score = val_error
        best_params = params

print("Best Parameters from Random Search:", best_params)

# ================================
# 3. Bayesian Optimization for Hyperparameter Tuning
# ================================
def objective(params):
    layers, learning_rate, momentum, activation = params
    nn = MyNeuralNetwork(
        layers=layers,
        learning_rate=learning_rate,
        momentum=momentum,
        activation=activation
    )
    loss = nn.fit(X_turbine, y_turbine, epochs=30)
    return loss[-1][1]

# Define the search space
space = [
    Categorical([[X_turbine.shape[1], 8, 8, 1], [X_turbine.shape[1], 16, 16, 1], [X_turbine.shape[1], 32, 32, 1]], name="layers"),
    Real(0.0001, 0.1, name="learning_rate"),
    Real(0.8, 0.99, name="momentum"),
    Categorical(["sigmoid", "relu", "tanh"], name="activation"),
]

# Run Bayesian Optimization
result = gp_minimize(objective, space, n_calls=20, random_state=42)

print("Best Parameters from Bayesian Optimization:", result.x)

# Visualization of the best configuration's training error can be added based on loss tracking.
