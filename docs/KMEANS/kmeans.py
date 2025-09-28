import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans

plt.figure(figsize=(12, 10))

# Gerar dados de amostra
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),   # Cluster em torno do ponto (0,0)
    np.random.normal(5, 1, (100, 2)),   # Cluster em torno do ponto (5,5)
    np.random.normal(10, 1, (100, 2))   # Cluster em torno do ponto (10,10)
])

# Executar K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# Resultados do enredo (scatter plot)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='*', s=200, label='Centróides')

plt.title('Resultados de agrupamento K-Means')
plt.xlabel('Recurso 1')
plt.ylabel('Recurso 2')
plt.legend()

# Imprimir centros e inércia
print("Centróides finais:", kmeans.cluster_centers_)
print("Inércia (WCSS):", kmeans.inertia_)

# Exibir o enredo
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
