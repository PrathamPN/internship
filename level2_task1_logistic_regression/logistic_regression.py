import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, classification_report,
                             confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 1. Load and Preprocess
print("\n1. Loading and Preprocessing the Dataset")
df = pd.read_csv("dataset/churn-bigml-80.csv")
print(f"Dataset Shape: {df.shape}")
print(f"Target: 'Churn' (binary - True/False)")

# Encode binary categoricals
le = LabelEncoder()
for col in ["International plan", "Voice mail plan", "Churn"]:
    df[col] = le.fit_transform(df[col])

# One-hot encode State
df = pd.get_dummies(df, columns=["State"], drop_first=True)

X = df.drop(columns=["Churn"])
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set : {X_test.shape[0]} samples")
print(f"Class distribution (Churn): {dict(y.value_counts())}")

# 2. Train Logistic Regression Model
print("\n2. Training Logistic Regression Model")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully.")

# 3. Interpret Coefficients and Odds Ratio
print("\n3. Model Coefficients and Odds Ratios")
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0],
    "Odds Ratio": np.exp(model.coef_[0])
}).sort_values("Odds Ratio", ascending=False)

print("\nTop 10 features by Odds Ratio:")
print(coef_df.head(10).to_string(index=False))
print("\nInterpretation: Odds Ratio > 1 means the feature increases churn probability.")

# 4. Evaluate the Model
print("\n4. Evaluating the Model")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Churn Prediction)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("level2_task1_logistic_regression/roc_curve.png", dpi=150)
plt.close()
print("\nROC Curve saved as 'roc_curve.png'")
