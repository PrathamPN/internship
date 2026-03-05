import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print("\n1. Loading and Preprocessing the Dataset")
df = pd.read_csv("dataset/churn-bigml-80.csv")
print(f"Dataset Shape: {df.shape}")

le = LabelEncoder()
for col in ["International plan", "Voice mail plan", "Churn"]:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=["State"], drop_first=True)

X = df.drop(columns=["Churn"])
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set : {X_test.shape[0]} samples")

print("\n2. Training Default Random Forest")
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
acc_default = accuracy_score(y_test, y_pred_default)
print(f"Default RF Accuracy: {acc_default:.4f}")

print("\n3. Hyperparameter Tuning with GridSearchCV")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

print("\n4. Evaluating Best Model on Test Set")
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

acc  = accuracy_score(y_test, y_pred_best)
prec = precision_score(y_test, y_pred_best)
rec  = recall_score(y_test, y_pred_best)
f1   = f1_score(y_test, y_pred_best)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

cv_scores = cross_val_score(best_rf, X_scaled, y, cv=5, scoring='f1')
print(f"\n5-Fold Cross-Validation F1-Scores: {cv_scores.round(4)}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))

print("\n5. Feature Importance Analysis")
importances = best_rf.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nTop 15 Most Important Features:")
print(feat_imp_df.head(15).to_string(index=False))

top15 = feat_imp_df.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top15["Feature"][::-1], top15["Importance"][::-1], color='steelblue')
plt.xlabel("Feature Importance (Mean Decrease in Impurity)")
plt.title("Random Forest - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("level3_task1_random_forest/feature_importance.png", dpi=150)
plt.close()
print("\nFeature importance chart saved as 'feature_importance.png'")
