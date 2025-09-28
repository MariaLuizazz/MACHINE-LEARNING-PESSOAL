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
