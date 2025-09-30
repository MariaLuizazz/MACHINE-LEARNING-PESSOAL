import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd

# Carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/bligui/Machine-Learning-Projetos/refs/heads/main/base/Obesity%20Classification.csv')

# Transformar variável "Gender" em numérica (Female=0, Male=1)
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Features (remover apenas a target "Label")
X = df.drop(columns=['Label'])

# Escalar dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redução de dimensionalidade PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Adicionar clusters ao DataFrame original
df['Cluster'] = labels

# Visualização dos clusters
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='Centróides')
plt.title('Clusters após redução de dimensionalidade (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

# Tabela de variância explicada
variancias = pca.explained_variance_ratio_
tabela_variancia = pd.DataFrame({
    'Componente Principal': [f'PC{i+1}' for i in range(len(variancias))],
    'Variância Explicada': variancias,
    'Variância Acumulada': np.cumsum(variancias)
})

print("\nTabela de Variância Explicada:")
print(tabela_variancia.to_markdown(index=False))

# Variância total explicada
print("\nVariância total explicada (2 componentes):", round(np.sum(variancias), 4))

# Resultados do KMeans em formato de tabela
tabela_centroides = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2'])
print("\nCentróides finais (no espaço PCA):")
print(tabela_centroides.to_markdown(index=False))

print("\nInércia (WCSS):", round(kmeans.inertia_, 2))

# Exportar gráfico em buffer
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
