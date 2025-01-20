import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load Dataset
file_path = 'Preprocessed/Preprocessed_A1_Synthetic.csv'  # Adjust path as necessary
df = pd.read_csv(file_path)

# Features and Target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Split Dataset (80% Train + Validation, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the BP-F Model
def build_bp_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Regression output
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

bp_model = build_bp_model(X_train.shape[1])

# Train the BP-F Model
history = bp_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predictions with BP-F
y_pred_bp = bp_model.predict(X_test).flatten()

# Metrics for BP-F
mse_bp = mean_squared_error(y_test, y_pred_bp)
mae_bp = mean_absolute_error(y_test, y_pred_bp)
mape_bp = np.mean(np.abs((y_test - y_pred_bp) / y_test)) * 100
r2_bp = r2_score(y_test, y_pred_bp)

print("BP-F Results:")
print(f"MSE: {mse_bp}, MAE: {mae_bp}, MAPE: {mape_bp}%, R2: {r2_bp}")

# Scatter Plot for BP-F
plt.scatter(y_test, y_pred_bp, alpha=0.5)
plt.title("BP-F: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

# Training and Validation Loss for BP-F
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("BP-F: Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()
