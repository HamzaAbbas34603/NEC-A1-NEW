import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load Dataset
file_path = './Preprocessed/Preprocessed_A1_synthetic.csv'   
df = pd.read_csv(file_path)

# Features and   Target
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

# Model Training
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred_mlr = mlr_model.predict(X_test)

# Metrics for MLR-F
mse_mlr = mean_squared_error(y_test, y_pred_mlr)
mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
mape_mlr = np.mean(np.abs((y_test - y_pred_mlr) / y_test)) * 100
r2_mlr = r2_score(y_test, y_pred_mlr)

print("MLR-F Results:")
print(f"MSE: {mse_mlr}, MAE: {mae_mlr}, MAPE: {mape_mlr}%, R2: {r2_mlr}")

# Scatter Plot for MLR-F
plt.scatter(y_test, y_pred_mlr, alpha=0.5)
plt.title("MLR-F: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

