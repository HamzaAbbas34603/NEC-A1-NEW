import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load the dataset
file_path = 'Preprocessed/preprocessed_Adversting_data.csv'
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column as the target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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
