import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# loading and preprocessing data
print("\n1. Loading and Preprocessing Data")

file_path = "dataset/1) iris.csv"
df = pd.read_csv(file_path)

print(f"Dataset Shape: {df.shape}")
print(f"Target Variable to Predict: 'species'")
print("\nFirst 3 rows:")
print(df.head(3))

print("\nClass Distribution:")
print(df['species'].value_counts())

if df.isnull().sum().sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values found. Dropping rows...")
    df.dropna(inplace=True)

X = df.drop(columns=['species'])
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size : {X_test.shape[0]} samples")

# training and evaluating the model (K=5)
print("\n2. Training and Evaluating Model (K=5)")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nAccuracy (K=5): {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
labels = sorted(y.unique())
conf_df = pd.DataFrame(conf_matrix, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
print(conf_df)

print("\nClassification Report (Precision/Recall/F1):")
print(class_report)

# comparing different values of K
print("\n3. Exploring Different Values of K")
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []

for k in k_values:
    temp_knn = KNeighborsClassifier(n_neighbors=k)
    temp_knn.fit(X_train, y_train)
    temp_pred = temp_knn.predict(X_test)
    temp_acc = accuracy_score(y_test, temp_pred)
    accuracies.append(temp_acc)

k_comparison = pd.DataFrame({'K Value': k_values, 'Accuracy': accuracies})
k_comparison['Accuracy'] = k_comparison['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
print(k_comparison.to_string(index=False))
