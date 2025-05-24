import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder  # Import OneHotEncoder
from sklearn.impute import SimpleImputer #Import SimpleImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

file_path = 'House Price Prediction Dataset.csv'
df = pd.read_csv(file_path)

# Handle Missing Values
#    - Strategy:  Use median for numerical, most frequent for categorical
#    - Rationale: Median is robust to outliers, frequent is suitable for categorical
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Create imputers
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Fit and transform the columns
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])


#Encode Categorical Variables
#    - Strategy: One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', drop='first')  #handle_unknown='ignore' added
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate the encoded columns with the original dataframe
df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Drop ID
#     - Rationale:  ID is not a predictive feature
df = df.drop('Id', axis=1)

# 5. Separate Features and Target, then Split
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print shapes to validate
# print(f"Shape of X_train: {X_train.shape}")
# print(f"Shape of X_test: {X_test.shape}")
# print(f"Shape of y_train: {y_train.shape}")
# print(f"Shape of y_test: {y_test.shape}")


# 5.  Exploratory Data Analysis (EDA)
#     - Rationale:  Understand data distribution, relationships, and identify potential issues

# 5.1 Summary Statistics
# print("\n--- Summary Statistics ---")
# print(df.describe())

# 5.2 Distribution of Price (Target Variable)
# plt.figure(figsize=(8, 6))
# sns.histplot(df['Price'], kde=True)
# plt.title('Distribution of House Prices')
# plt.show()

# 5.3 Correlation Heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

# 5.4 Scatter Plots (Numerical Features vs. Price)
# for col in numerical_cols:
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=df[col], y=df['Price'])
#     plt.title(f'{col} vs. Price')
#     plt.show()


# 6. Separate Features and Target, then Split
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# print shapes to validate
# print(f"Shape of X_train: {X_train.shape}")
# print(f"Shape of X_test: {X_test.shape}")
# print(f"Shape of y_train: {y_train.shape}")
# print(f"Shape of y_test: {y_test.shape}")

# Initialize XGBoost model with careful parameters to prevent overfitting
xgb_model = XGBRegressor(
    n_estimators=200,       # Number of boosting rounds
    max_depth=5,           # Maximum tree depth
    learning_rate=0.1,     # Shrinkage factor
    subsample=0.8,         # Fraction of samples used per tree
    colsample_bytree=0.8,  # Fraction of features used per tree
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_train_xgb = xgb_model.predict(X_train)
y_pred_test_xgb = xgb_model.predict(X_test)

# Evaluate model performance
def evaluate_model(y_true, y_pred, set_name):
    print(f"\nXGBoost Evaluation metrics for {set_name}:")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_train, y_pred_train_xgb, "Training Set")
evaluate_model(y_test, y_pred_test_xgb, "Test Set")

# Feature importance
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importances.head(10))

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test_xgb)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('XGBoost: Actual vs Predicted House Prices (Test Set)')
plt.show()

# Residual plot
residuals_xgb = y_test - y_pred_test_xgb
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_test_xgb, y=residuals_xgb)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('XGBoost: Residual Plot')
plt.show()