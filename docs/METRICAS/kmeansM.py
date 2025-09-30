import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    silhouette_score
)

# ==============================
# Carregamento dos dados
# ==============================
df = pd.read_csv("https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv")

X = df.drop(columns=["diagnosis", "id"])
y_true = df["diagnosis"].map({"M": 1, "B": 0})  # ground truth para avaliar

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans
kmeans = KMeans(n_clusters=2, init="k-means++", max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# ==============================
# Métricas de avaliação
# ==============================
ari = adjusted_rand_score(y_true, labels)
ami = adjusted_mutual_info_score(y_true, labels)
hom = homogeneity_score(y_true, labels)
comp = completeness_score(y_true, labels)
vmes = v_measure_score(y_true, labels)
sil = silhouette_score(X_pca, labels)

print("Métricas KMeans")
print(f"ARI:           {ari:.4f}")
print(f"AMI:           {ami:.4f}")
print(f"Homogeneidade: {hom:.4f}")
print(f"Completude:    {comp:.4f}")
print(f"V-Measure:     {vmes:.4f}")
print(f"Silhouette:    {sil:.4f}")

# ==============================
# Visualização
# ==============================
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="*", s=200, label="Centróides")
plt.title("Clusters após PCA - KMeans")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
