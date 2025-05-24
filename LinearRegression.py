import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

#scaling
numerical_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


X_train, X_test, y_train, y_test = train_test_split(
    df_encoded.drop('Price', axis=1),
    df_encoded['Price'],
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

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
corr_matrix = df_encoded[numerical_cols + ['Price']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()

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

# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

# Evaluate model performance
def evaluate_model(y_true, y_pred, set_name):
    print(f"\nEvaluation metrics for {set_name}:")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_train, y_pred_train, "Training Set")
evaluate_model(y_test, y_pred_test, "Test Set")

# Feature importance (coefficients)
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nTop 10 most important features:")
print(coefficients.head(10))

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (Test Set)')
plt.show()

# Residual plot
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
