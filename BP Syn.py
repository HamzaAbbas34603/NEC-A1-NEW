import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from backpropagation import MyNeuralNetwork

# Load the Boston Housing dataset
Synthetic_data_path = 'normalized_A1_synthetic.csv'
Synthetic_data = pd.read_csv(Synthetic_data_path)

# Normalize the data using Min-Max scaling
min_max_scaler = MinMaxScaler()
Synthetic_data_normalized = pd.DataFrame(min_max_scaler.fit_transform(Synthetic_data), columns=Synthetic_data.columns)

# Extract features (X) and target variable (y)
X = Synthetic_data_normalized.drop('z', axis=1).values
y = Synthetic_data_normalized['z'].values


class MyNeuralNetwork:
    def __init__(self, layers, learning_rate, momentum, activation, validation_percentage=0):
        self.L = len(layers)
        self.n = layers.copy()
        self.xi = [np.zeros(l) for l in layers]
        self.w = [np.random.randn(layers[i], layers[i-1]) for i in range(1, self.L)]
        self.theta = [np.zeros(l) for l in layers]
        self.delta = [np.zeros(l) for l in layers]
        self.d_w = [np.zeros_like(w) for w in self.w]
        self.d_theta = [np.zeros_like(t) for t in self.theta]
        self.d_w_prev = [np.zeros_like(w) for w in self.w]
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation = activation
        self.validation_percentage = validation_percentage
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
            self.d_w[i-1] = self.learning_rate * np.outer(self.delta[i], self.xi[i-1]) + self.momentum * self.d_w_prev[i-1]
            self.d_theta[i] = self.learning_rate * self.delta[i] + self.momentum * self.d_theta_prev[i]
            self.w[i-1] -= self.d_w[i-1]
            self.theta[i] -= self.d_theta[i]
            self.d_w_prev[i-1] = self.d_w[i-1]
            self.d_theta_prev[i] = self.d_theta[i]

    def fit(self, X, y, epochs, batch_size=32, learning_rate_decay=0.1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_percentage, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        loss_epochs = []

        for epoch in range(epochs):
            for _ in range(0, len(X_train), batch_size):
                batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
                X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                for i in range(len(X_batch)):
                    x_sample, y_sample = X_batch[i], y_batch[i]
                    self.forward_pass(x_sample)
                    self.backward_pass(y_sample)
                    self.update_weights()

                # Calculate training error
                train_predictions = self.predict(X_train)
                training_error = np.mean((train_predictions - y_train) ** 2)

                # Calculate validation error
                val_predictions = self.predict(X_val)
                validation_error = np.mean((val_predictions - y_val) ** 2)

                self.loss_epochs.append([training_error, validation_error])

                # Print values during training
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - Training Error: {training_error}, Validation Error: {validation_error}")

            # Learning rate decay
            self.learning_rate *= (1.0 / (1.0 + learning_rate_decay * epoch))

        return np.array(self.loss_epochs)

    def predict(self, X):
        # Transform input data using the same scaler as used during training
        X = StandardScaler().fit_transform(X)
        predictions = []
        for i in range(len(X)):
            self.forward_pass(X[i])
            predictions.append(self.xi[-1][0])
        return np.array(predictions)

    def get_loss_epochs(self):
        return np.array(self.loss_epochs)

# Define neural network parameters with two hidden layers
layers = [9, 10, 8, 1]  # Input layer: 13 features, Hidden layers: 10, 8, Output layer: 1 unit
learning_rate = 0.01
momentum = 0.9
activation = 'sigmoid'
validation_percentage = 0.2
epochs = 100

# Create and train the neural network
nn = MyNeuralNetwork(layers, learning_rate, momentum, activation, validation_percentage)
loss_history = nn.fit(X, y, epochs)

# Plot the evolution of the training and validation errors
import matplotlib.pyplot as plt

plt.plot(loss_history[:, 0], label='Training Error')
plt.plot(loss_history[:, 1], label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Scatter plot with diagonal line and separate colors for predicted and actual values
plt.figure(figsize=(10, 6))
predictions = nn.predict(X)
plt.scatter(y, predictions, color='red', label='Predicted Values', alpha=0.5)
plt.scatter(y, y, color='blue', label='Actual Values', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Predicted vs Actual Values')
plt.legend()
plt.show()