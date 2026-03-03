import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# 1. Load and Preprocess
print("\n1. Loading and Preprocessing the Dataset")
df = pd.read_csv("dataset/churn-bigml-80.csv")
print(f"Dataset Shape: {df.shape}")

# Use only numerical features for clustering
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical features used for clustering: {len(numerical_cols)}")

X = df[numerical_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized.")

# 2. Elbow Method to Find Optimal K
print("\n2. Finding Optimal Number of Clusters (Elbow Method)")
inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  K={k}: Inertia = {km.inertia_:.2f}")

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method - Optimal K for Customer Segmentation")
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("level2_task3_kmeans/elbow_plot.png", dpi=150)
plt.close()
print("\nElbow plot saved as 'elbow_plot.png'")
print("Based on the elbow curve, K=3 is chosen as the optimal number of clusters.")

# 3. Apply K-Means with Optimal K=3
print("\n3. Applying K-Means with K=3")
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters
print(f"Cluster assignment complete.")
print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index().to_string())

# 4. Visualize Clusters using PCA (2D projection)
print("\n4. Visualizing Clusters in 2D (PCA)")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
variance_explained = pca.explained_variance_ratio_.sum() * 100
print(f"PCA explains {variance_explained:.1f}% of total variance")

colors = ['#e74c3c', '#3498db', '#2ecc71']
labels = [f"Cluster {i}" for i in range(optimal_k)]

plt.figure(figsize=(9, 6))
for i in range(optimal_k):
    mask = clusters == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=labels[i], alpha=0.6, s=40)

# Plot cluster centers on PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='black', marker='X', s=200, label='Centroids', zorder=5)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title(f"K-Means Clustering (K=3) - Customer Segmentation\n(PCA 2D Projection, {variance_explained:.1f}% variance explained)")
plt.legend()
plt.tight_layout()
plt.savefig("level2_task3_kmeans/clusters.png", dpi=150)
plt.close()
print("Cluster scatter plot saved as 'clusters.png'")

# 5. Interpret Cluster Characteristics
print("\n5. Interpreting Cluster Characteristics")
cluster_summary = df.groupby("Cluster")[numerical_cols].mean()
print("\nCluster Mean Values (key features):")
key_features = ["Account length", "Total day minutes", "Total day charge",
                "Customer service calls", "Number vmail messages"]
print(cluster_summary[key_features].round(2).to_string())
