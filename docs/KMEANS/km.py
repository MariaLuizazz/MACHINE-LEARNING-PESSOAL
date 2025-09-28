import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# ----------------------------
# Importar dados
# ----------------------------
df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')

# Features (remover id e diagnóstico)
X = df.drop(columns=['diagnosis', 'id'])

# ----------------------------
# Normalização
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# PCA para reduzir para 2 dimensões
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ----------------------------
# Executar K-Means
# ----------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Adicionar clusters ao dataframe
df['Cluster'] = labels

# ----------------------------
# Plot dos clusters
# ----------------------------
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='Centróides')
plt.title('Clusters após redução de dimensionalidade (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

# ----------------------------
# Tabela de variância explicada
# ----------------------------
variancias = pca.explained_variance_ratio_
tabela_variancia = pd.DataFrame({
    'Componente Principal': [f'PC{i+1}' for i in range(len(variancias))],
    'Variância Explicada': variancias,
    'Variância Acumulada': np.cumsum(variancias)
})

# Exibir tabela em Markdown (perfeito para MkDocs)
print(tabela_variancia.to_markdown(index=False))

# Variância total
print("\nVariância total explicada (2 componentes):", np.sum(variancias))

# ----------------------------
# Centròides e inércia
# ----------------------------
print("\nCentróides finais:", kmeans.cluster_centers_)
print("Inércia (WCSS):", kmeans.inertia_)

# ----------------------------
# Exibir enredo em SVG
# ----------------------------
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
