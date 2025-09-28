import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Importar dados
df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')

# Features (remover id e diagnóstico)
X = df.drop(columns=['diagnosis', 'id'])

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para reduzir para 3 dimensões
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Executar K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Adicionar os clusters ao dataframe
df['Cluster'] = labels

# Plot 3D dos clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Pontos
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis', s=50)

# Centròides
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
           c='red', marker='*', s=200, label='Centróides')

ax.set_title('Clusters após redução de dimensionalidade (PCA 3D)')
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.legend()
plt.show()

# Mostrar variância explicada
print("Variância explicada por cada componente:", pca.explained_variance_ratio_)
print("Variância total explicada (3 componentes):", np.sum(pca.explained_variance_ratio_))

# Imprimir centros e inércia
print("Centróides finais:", kmeans.cluster_centers_)
print("Inércia (WCSS):", kmeans.inertia_)
