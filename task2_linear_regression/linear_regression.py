import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# loading and preprocessing data
print("\n1. Loading and Preprocessing Data")

file_path = "dataset/4) house Prediction Data Set.csv"

column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

df = pd.read_csv(file_path, sep=r'\s+', names=column_names)

print(f"Dataset Shape: {df.shape}")
print(f"Target Variable to Predict: 'MEDV' (Median House Value in $1000s)")
print("\nFirst 3 rows:")
print(df.head(3))

if df.isnull().sum().sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values found. Dropping rows...")
    df.dropna(inplace=True)

X = df.drop(columns=['MEDV'])
y = df['MEDV']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size : {X_test.shape[0]} samples")

# training the model
print("\n2. Training the Model")
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression model trained successfully.")

# interpreting model coefficients
print("\n3. Interpreting Model Coefficients")
print(f"Model Intercept (Base Value): {model.intercept_:.3f}")

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

coef_df['Abs_Magnitude'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Magnitude', ascending=False).drop(columns=['Abs_Magnitude'])

print("\nFeatures sorted by impact on prediction (per 1 standard deviation change):")
print(coef_df.to_string(index=False))

print("\nInterpretation Example:")
top_feature = coef_df.iloc[0]['Feature']
top_coef = coef_df.iloc[0]['Coefficient']
direction = "increase" if top_coef > 0 else "decrease"
print(f"-> A 1 std deviation increase in '{top_feature}' leads to a {abs(top_coef):.3f} point {direction} in predicted house price.")

# evaluating the model
print("\n4. Evaluating the Model")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Sq Error (RMSE): {rmse:.3f} ($1000s)")
print(f"R-squared (R2 Score)    : {r2:.3f} (Model explains {r2*100:.1f}% of variance)")
