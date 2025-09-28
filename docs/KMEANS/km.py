import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate  

plt.figure(figsize=(12, 10))

df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')

#features tira id e diagnostico
X = df.drop(columns=['diagnosis', 'id'])

#normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Executar K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels



# Resultados do enredo (scatter plot)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
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
