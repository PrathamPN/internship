import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# 1. Load and Preprocess
# Use Iris with 2 classes for clear binary comparison: versicolor vs virginica
print("\n1. Loading and Preprocessing the Dataset")
df = pd.read_csv("dataset/1) iris.csv")
df = df[df['species'].isin(['versicolor', 'virginica'])].copy()
print(f"Dataset: Iris (2-class: versicolor vs virginica)")
print(f"Dataset Shape: {df.shape}")

le = LabelEncoder()
X = df.drop(columns=["species"])
y = le.fit_transform(df["species"])   # versicolor=0, virginica=1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set : {X_test.shape[0]} samples")
print(f"Classes: {le.classes_} -> [0, 1]")

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.decision_function(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_prob)
    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))
    return model, acc, prec, rec, auc

# 2. Train SVM - Linear Kernel
print("\n2. Training SVM with Linear Kernel")
svm_linear = SVC(kernel='linear', probability=False, random_state=42)
svm_linear, acc_l, prec_l, rec_l, auc_l = evaluate(
    "Linear SVM", svm_linear, X_train, y_train, X_test, y_test)

# 3. Train SVM - RBF Kernel
print("\n3. Training SVM with RBF Kernel")
svm_rbf = SVC(kernel='rbf', probability=False, random_state=42)
svm_rbf, acc_r, prec_r, rec_r, auc_r = evaluate(
    "RBF SVM", svm_rbf, X_train, y_train, X_test, y_test)

# 4. Compare Results
print("\n4. Comparison: Linear vs RBF Kernel")
comp = pd.DataFrame({
    "Metric"   : ["Accuracy", "Precision", "Recall", "AUC"],
    "Linear SVM": [f"{acc_l:.4f}", f"{prec_l:.4f}", f"{rec_l:.4f}", f"{auc_l:.4f}"],
    "RBF SVM"   : [f"{acc_r:.4f}", f"{prec_r:.4f}", f"{rec_r:.4f}", f"{auc_r:.4f}"]
})
print(comp.to_string(index=False))

# 5. Visualize Decision Boundary using first 2 PCA components
print("\n5. Visualizing Decision Boundaries (PCA 2D projection)")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_tr_pca, X_te_pca, y_tr_pca, y_te_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#3498db', '#e74c3c']
class_names = list(le.classes_)

for ax, (name, kernel) in zip(axes, [("Linear", 'linear'), ("RBF", 'rbf')]):
    svm_2d = SVC(kernel=kernel, random_state=42)
    svm_2d.fit(X_tr_pca, y_tr_pca)

    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, colors=colors)
    for cls, color, label in zip([0, 1], colors, class_names):
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label,
                   s=50, edgecolors='k', linewidth=0.5)
    ax.set_title(f"SVM Decision Boundary ({name} Kernel)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

plt.suptitle("SVM Decision Boundaries - versicolor vs virginica (PCA 2D)", fontsize=13)
plt.tight_layout()
plt.savefig("level3_task2_svm/decision_boundary.png", dpi=150)
plt.close()
print("Decision boundary saved as 'decision_boundary.png'")
