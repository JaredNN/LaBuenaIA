import os
import sys
import warnings
import numpy as np
import matplotlib
# matplotlib.use('Agg') # Forzamos trabajo en segundo plano sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.cluster import KMeans

# Ignorar advertencias visuales para mantener la terminal limpia
warnings.filterwarnings('ignore')

# 1. Configuración del entorno y carpetas (Estilo Profesional)
script_dir = Path.cwd()
output_folder = script_dir / 'resultados_kmeans_profesional'
output_folder.mkdir(exist_ok=True)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 60)
print("EJECUCIÓN DE K-MEANS: FLUJO PROFESIONAL Y GUARDADO AUTOMÁTICO")
print("=" * 60)

# 2. Generación de Datos
print("\n[1/4] Generando datos de ejemplo...")
np.random.seed(42)
cluster1 = np.random.randn(50, 2) + [2, 2]
cluster2 = np.random.randn(50, 2) + [8, 8]
cluster3 = np.random.randn(50, 2) + [2, 8]
X = np.vstack([cluster1, cluster2, cluster3])

print(f"✓ Datos generados: 3 grupos, total de {X.shape[0]} muestras.")

# 3. Entrenamiento del Modelo K-Means
print("\n[2/4] Entrenando modelo K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_
print("✓ Entrenamiento completado exitosamente.")

# 4. Análisis de Resultados
print("\n[3/4] Analizando resultados de los clusters...")
print("Distribución de puntos:")
for i in range(3):
    print(f"   - Cluster {i+1}: {np.sum(labels == i)} puntos")

print("\nCoordenadas exactas de los centroides:")
for i, center in enumerate(centers):
    print(f"   - Centroide {i+1}: X={center[0]:.2f}, Y={center[1]:.2f}")

# 5. Visualización Avanzada y Guardado Automático
print("\n[4/4] Generando y guardando la gráfica de clusters...")

fig, ax = plt.subplots()

# Paleta de colores atractiva
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

# Graficar los puntos de cada cluster
for i in range(3):
    cluster_points = X[labels == i]
    ax.scatter(
        cluster_points[:, 0], cluster_points[:, 1], 
        c=colors[i], label=f'Cluster {i+1}', 
        s=100, alpha=0.6, edgecolors='black', linewidth=1.5
    )

# Graficar los centroides
ax.scatter(
    centers[:, 0], centers[:, 1], 
    c='red', marker='X', s=300, 
    label='Centroides', edgecolors='black', 
    linewidth=2, zorder=10
)

# Configuración estética de la gráfica
ax.set_title('Clustering K-Means con 3 Clusters', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Característica 1', fontsize=12)
ax.set_ylabel('Característica 2', fontsize=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Guardado automático en la carpeta generada
ruta_imagen = output_folder / 'kmeans_visualization_optimo.png'
plt.savefig(ruta_imagen, dpi=300)

print(f"✓ Gráfica guardada exitosamente en: {ruta_imagen.absolute()}")
print("-> Abriendo gráfica en una ventana nueva...")
plt.show()
print("\n================ PROCESO TERMINADO ================")