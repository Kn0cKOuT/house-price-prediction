import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

file_path = 'House Price Prediction Dataset.csv'
df = pd.read_csv(file_path)

# Separate features (X) and target (y)
X = df.drop(['Price', 'Id'], axis=1)
y = df['Price']



# Fill missing numerical values with median
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 1. Remove the ID column
X = df_encoded.drop(['Price', 'Id'], axis=1)


X = df_encoded.drop(['Price', 'Id'], axis=1)
y = df_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# --------------------------------------------------------------------------------------

# Exploratory Data Analysis

# Summary stats for numerical features
# print("Summary stats for numerical features:")
# print(df_encoded[numerical_cols].describe())
#
# # Summary stats for categorical (encoded) features
# print("Summary stats for categorical (encoded) features:")
categorical_encoded_cols = [col for col in df_encoded.columns if col.startswith(tuple(categorical_cols))]
# print(df_encoded[categorical_encoded_cols].describe())

# Distribution of Price
# plt.figure(figsize=(10, 6))
# sns.histplot(y, kde=True, bins=30)
# plt.title("Distribution of House Prices")
# plt.xlabel("Price")
# plt.show()

# Summary stats
# print(y.describe())

# # Correlation matrix (numerical features + target)
# corr_matrix = df_encoded[numerical_cols + ['Price']].corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title("Correlation Heatmap")
# plt.show()

# # Scatter plots for numerical features
# for col in numerical_cols:
# #     plt.figure(figsize=(8, 4))
# #     sns.scatterplot(x=df_encoded[col], y=y)
# #     plt.title(f"{col} vs Price")
# #     plt.show()

# # Boxplots for categorical features
# for col in categorical_encoded_cols:
# #     plt.figure(figsize=(8, 4))
# #     sns.boxplot(x=df_encoded[col], y=y)
# #     plt.title(f"{col} vs Price")
# #     plt.show()

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees in the forest
    random_state=42,   # For reproducibility
    max_depth=None,    # Let trees grow fully (can tune this later)
    min_samples_split=2,
    n_jobs=-1         # Use all CPU cores
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

# Evaluate model performance
def evaluate_model(y_true, y_pred, set_name):
    print(f"\nRandom Forest Evaluation metrics for {set_name}:")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_train, y_pred_train_rf, "Training Set")
evaluate_model(y_test, y_pred_test_rf, "Test Set")

# Feature importance
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importances.head(10))

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest: Actual vs Predicted House Prices (Test Set)')
plt.show()

# Residual plot
residuals_rf = y_test - y_pred_test_rf
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_test_rf, y=residuals_rf)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Random Forest: Residual Plot')
plt.show()