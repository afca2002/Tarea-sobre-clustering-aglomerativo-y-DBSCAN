import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Datos proporcionados
data = np.array([
    [1, 2], [1, 3], [2, 2], [2, 3],
    [4, 6], [8, 8], [8, 9], [8, 5], [8, 7], [7, 6]
])

# Aplicar DBSCAN
dbscan = DBSCAN(eps=2, min_samples=2)
clusters = dbscan.fit_predict(data)

# Identificar outliers (cluster = -1)
outliers = data[clusters == -1]

# Calcular métrica de silueta (si hay al menos 2 clusters)
unique_clusters = np.unique(clusters)
if len(unique_clusters) > 1 and -1 in unique_clusters:
    sample_data = data[clusters != -1]
    sample_labels = clusters[clusters != -1]
    if len(np.unique(sample_labels)) > 1:
        silhouette = silhouette_score(sample_data, sample_labels)
        print(f"Coeficiente de silueta (sin outliers): {silhouette:.2f}")

# Visualización mejorada
plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

for cluster, color in zip(unique_clusters, colors):
    if cluster == -1:
        plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', s=100, label='Outliers')
    else:
        cluster_points = data[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f'Cluster {cluster}')

plt.title('Resultados de DBSCAN (ε=2, min_samples=2)', fontsize=14)
plt.xlabel('Coordenada X', fontsize=12)
plt.ylabel('Coordenada Y', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()