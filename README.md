# 🏠 House Price Prediction

This project aims to predict house prices using multiple machine learning models. It compares and evaluates the performance of the following algorithms:

- 🔹 Linear Regression
- 🌲 Random Forest Regressor
- 🚀 XGBoost Regressor

## 📁 Dataset

The dataset used in this project is named `House Price Prediction Dataset.csv`. It contains both numerical and categorical features such as area, number of bedrooms/bathrooms, floors, year built, and location-related attributes.

## ⚙️ Technologies Used

- Python (pandas, scikit-learn, seaborn, matplotlib, xgboost)
- Jupyter Notebook / Python Scripts for analysis and modeling
- Machine Learning Models:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `XGBRegressor`

## 📊 Exploratory Data Analysis (EDA)

The dataset has been explored and prepared through:
- Handling missing values (median for numerical, most frequent for categorical)
- One-hot encoding for categorical variables
- Correlation analysis using heatmaps
- Price distribution visualization

## 🧠 Model Training & Evaluation

Each model is evaluated using the following metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of Determination)

Predictions are visualized and compared against actual values using scatter plots and residual plots.

## 📌 File Descriptions

| File Name                  | Description                                                          |
|---------------------------|----------------------------------------------------------------------|
| `LinearRegression.py`     | Training and evaluating a linear regression model                    |
| `RandomForestRegressor.py`| Training and evaluating a random forest model + feature importance   |
| `XGBoostRegressor.py`     | Training and evaluating an XGBoost model with hyperparameters        |
| `House Price Prediction Dataset.csv` | The dataset used for training and testing                 |

## 📈 Results

- **XGBoost** showed the best performance in terms of accuracy and generalization.
- **Random Forest** offered better interpretability through feature importance.
- **Linear Regression** served as a baseline model.

## 🔧 Requirements

To run the project, install the necessary libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
