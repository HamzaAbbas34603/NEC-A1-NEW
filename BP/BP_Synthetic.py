# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# ===============================
# 1. Load and Preprocess Datasets
# ================================

# Load datasets
synthetic_data = pd.read_csv('./Preprocessed/Preprocessed_A1_synthetic.csv', sep=',', header=0)

# Display basic info for sanity checks
print("\nSynthetic Data Info:")
synthetic_data.info()

# Extract features (X) and target (y) for each dataset
X_synthetic = synthetic_data.drop('z', axis=1).values
y_synthetic = synthetic_data['z'].values

# ================================
# 2. Define Neural Network Class
# ================================

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
# 3. Train and Evaluate Models
# ================================
layers_synthetic = [9, 20, 25, 1]  # For synthetic data
learning_rate = 0.001
momentum = 0.9
activation = 'sigmoid'
epochs = 50

# Create and train the models for each dataset
nn_synthetic = MyNeuralNetwork(layers_synthetic, learning_rate, momentum, activation)
loss_synthetic = nn_synthetic.fit(X_synthetic, y_synthetic, epochs)

# ================================
# 4. Evaluation and Visualization
# ================================

# Predict on the dataset
y_pred = nn_synthetic.predict(X_synthetic)

# Calculate evaluation metrics
mse = mean_squared_error(y_synthetic, y_pred)
mae = mean_absolute_error(y_synthetic, y_pred)
mape = mean_absolute_percentage_error(y_synthetic, y_pred)
r2 = r2_score(y_synthetic, y_pred)

print(f"Evaluation Metrics:\nMSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2 Score: {r2:.4f}")

# Plot loss
def plot_loss(loss, title):
    plt.plot(loss[:, 0], label='Train Error')
    plt.plot(loss[:, 1], label='Validation Error')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

plot_loss(loss_synthetic, "Synthetic Data Loss")

# Plot actual vs predicted
def plot_actual_vs_predicted(y_actual, y_predicted):
    plt.scatter(y_actual, y_predicted, alpha=0.7)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', linestyle='--')
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

plot_actual_vs_predicted(y_synthetic, y_pred)
