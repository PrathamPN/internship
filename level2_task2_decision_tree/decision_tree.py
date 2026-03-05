import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("\n1. Loading and Preprocessing the Dataset")
df = pd.read_csv("dataset/1) iris.csv")
print(f"Dataset Shape: {df.shape}")
print(f"Classes: {df['species'].unique()}")

X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set : {X_test.shape[0]} samples")

print("\n2. Training Full Decision Tree (No Pruning)")
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

y_pred_full = dt_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)
f1_full = f1_score(y_test, y_pred_full, average='weighted')

print(f"Tree Depth      : {dt_full.get_depth()}")
print(f"Number of Leaves: {dt_full.get_n_leaves()}")
print(f"Accuracy        : {acc_full:.4f}")
print(f"F1-Score        : {f1_full:.4f}")

print("\n3. Visualizing Full Tree")
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(dt_full, feature_names=X.columns, class_names=dt_full.classes_,
          filled=True, rounded=True, ax=ax, fontsize=8)
plt.title("Full Decision Tree - Iris Dataset")
plt.tight_layout()
plt.savefig("level2_task2_decision_tree/decision_tree_full.png", dpi=150)
plt.close()
print("Full tree saved as 'decision_tree_full.png'")

print("\n4. Pruning the Tree (max_depth=3)")
dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(X_train, y_train)

y_pred_pruned = dt_pruned.predict(X_test)
acc_pruned = accuracy_score(y_test, y_pred_pruned)
f1_pruned = f1_score(y_test, y_pred_pruned, average='weighted')

print(f"Tree Depth      : {dt_pruned.get_depth()}")
print(f"Number of Leaves: {dt_pruned.get_n_leaves()}")
print(f"Accuracy        : {acc_pruned:.4f}")
print(f"F1-Score        : {f1_pruned:.4f}")

fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(dt_pruned, feature_names=X.columns, class_names=dt_pruned.classes_,
          filled=True, rounded=True, ax=ax, fontsize=9)
plt.title("Pruned Decision Tree (max_depth=3) - Iris Dataset")
plt.tight_layout()
plt.savefig("level2_task2_decision_tree/decision_tree_pruned.png", dpi=150)
plt.close()
print("Pruned tree saved as 'decision_tree_pruned.png'")

print("\n5. Full vs Pruned Tree Comparison")
comparison = pd.DataFrame({
    "Metric"    : ["Depth", "Leaves", "Accuracy", "F1-Score (weighted)"],
    "Full Tree" : [dt_full.get_depth(), dt_full.get_n_leaves(),
                   f"{acc_full:.4f}", f"{f1_full:.4f}"],
    "Pruned (depth=3)": [dt_pruned.get_depth(), dt_pruned.get_n_leaves(),
                         f"{acc_pruned:.4f}", f"{f1_pruned:.4f}"]
})
print(comparison.to_string(index=False))

print("\nClassification Report (Pruned Tree):")
print(classification_report(y_test, y_pred_pruned))
