import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from mlxtend.plotting import plot_decision_regions

# Ignorar advertencias (común en redes neuronales por los límites de iteración)
warnings.filterwarnings('ignore')

# 1. Configuración de carpetas
script_dir = Path.cwd()
output_folder = script_dir / 'resultados_mlp_profesional'
output_folder.mkdir(exist_ok=True)

plt.style.use('ggplot')

print("=" * 60)
print("EJECUCIÓN DE MULTILAYER PERCEPTRON (MLP): FLUJO PROFESIONAL")
print("=" * 60)

# 2. Generación de Datos Complejos (Forma de Medias Lunas)
print("\n[1/4] Generando dataset no lineal (Make Moons)...")
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"✓ Datos listos. Entrenamiento: {len(X_train)} muestras. Prueba: {len(X_test)} muestras.")

# 3. Entrenamiento de la Red Neuronal (MLP)
# Arquitectura: 2 capas ocultas con 10 neuronas cada una. Función de activación ReLU.
print("\n[2/4] Entrenando la Red Neuronal Artificial...")
modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10), 
    activation='relu', 
    solver='adam', 
    max_iter=1000, 
    random_state=42
)
modelo_mlp.fit(X_train, y_train)
print("✓ Entrenamiento completado exitosamente.")

# 4. Evaluación del Modelo
print("\n[3/4] Evaluando la precisión de la red neuronal...")
predicciones = modelo_mlp.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)

print(f"✓ Accuracy final: {accuracy * 100:.2f}%\n")
print("Reporte de clasificación detallado:")
print(classification_report(y_test, predicciones))

# 5. Visualización Avanzada y Guardado
print("\n[4/4] Generando gráfica de fronteras de decisión (No lineal)...")
fig, ax = plt.subplots(figsize=(10, 6))

# Usamos mlxtend para dibujar cómo la red neuronal envuelve los datos
plot_decision_regions(
    X=X_train,
    y=y_train,
    clf=modelo_mlp,
    ax=ax,
    scatter_kwargs={'s': 50, 'edgecolor': 'black', 'alpha': 0.8}
)

# Configuración estética
ax.set_title(f'Red Neuronal (MLP) - Clasificación No Lineal\nAccuracy: {accuracy*100:.2f}% | Capas: (10, 10)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Característica 1', fontsize=12)
ax.set_ylabel('Característica 2', fontsize=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Mostrar la gráfica en una ventana interactiva
print("✓ Abriendo ventana con la gráfica...")
plt.show()

print("\n================ PROCESO TERMINADO ================")