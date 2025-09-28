import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
print(df.sample(n=10, random_state=42).to_markdown(index=False))

