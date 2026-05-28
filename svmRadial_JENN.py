import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
# Configuración (Código 2): Forzamos a matplotlib a trabajar en segundo plano sin abrir ventanas
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import style
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from mlxtend.plotting import plot_decision_regions

# Ignorar advertencias visuales para mantener la terminal limpia
warnings.filterwarnings('ignore')

# 1. Configuración del entorno y carpetas (Del Código 2)
script_dir = Path.cwd()
output_folder = script_dir / 'resultados_svm_profesional'
output_folder.mkdir(exist_ok=True)

style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 60)
print("EJECUCIÓN DE SVM: FLUJO PROFESIONAL Y GUARDADO AUTOMÁTICO")
print("=" * 60)

# 2. Carga de Datos (Del Código 1 - Entorno real)
print("\n[1/4] Descargando y preparando el dataset...")
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/master/data/ESL.mixture.csv'
datos = pd.read_csv(url)

X = datos.drop(columns='y')
y = datos['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1234, shuffle=True
)
print(f"✓ Datos listos. Entrenamiento: {X_train.shape[0]} muestras. Prueba: {X_test.shape[0]} muestras.")

# 3. Optimización Automática de Hiperparámetros (Del Código 1)
print("\n[2/4] Buscando la mejor configuración matemática (GridSearchCV)...")
param_grid = {
    'C': np.logspace(-2, 3, 6),
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

# GridSearchCV probará todas las combinaciones posibles internamente
grid_search = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,  # Utiliza todos los núcleos de tu procesador para ir más rápido
    cv=5,
    verbose=0
)

grid_search.fit(X_train, y_train)
modelo_final = grid_search.best_estimator_

print(f"✓ Mejor configuración encontrada:")
print(f"   - Parámetro C: {grid_search.best_params_['C']}")
print(f"   - Parámetro Gamma: {grid_search.best_params_['gamma']}")

# 4. Evaluación del Modelo
print("\n[3/4] Evaluando el modelo con datos de prueba...")
predicciones = modelo_final.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)

print(f"✓ Accuracy final: {accuracy * 100:.2f}%\n")
print("Reporte de clasificación detallado:")
print(classification_report(y_test, predicciones))

# 5. Visualización Avanzada (Combinación de ambos códigos)
print("\n[4/4] Generando y guardando la gráfica de fronteras de decisión...")
fig, ax = plt.subplots(figsize=(10, 6))

# Usamos mlxtend (del Código 1) porque dibuja las áreas de color de forma más elegante
plot_decision_regions(
    X=X_train.to_numpy(),
    y=y_train.to_numpy().astype(int), 
    clf=modelo_final,
    ax=ax,
    scatter_kwargs={'s': 40, 'edgecolor': 'black', 'alpha': 0.7}
)

# Resaltamos los vectores de soporte con aros amarillos (del Código 2)
ax.scatter(
    modelo_final.support_vectors_[:, 0],
    modelo_final.support_vectors_[:, 1],
    s=150, linewidth=2, facecolors='none', edgecolors='yellow',
    label='Vectores de Soporte'
)

# Formato final de la imagen
ax.set_title(f'Fronteras de Decisión (SVM Radial)\nAccuracy Test: {accuracy*100:.2f}% | C={grid_search.best_params_["C"]:.2f}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Característica X1', fontsize=11)
ax.set_ylabel('Característica X2', fontsize=11)
ax.legend(loc='upper left')

# Guardado automático y mostrar en pantalla
ruta_imagen = output_folder / 'svm_fronteras_decision_optimo.png'
plt.savefig(ruta_imagen, dpi=300)

print(f"✓ Gráfica guardada exitosamente en: {ruta_imagen.absolute()}")
print("-> Abriendo gráfica en una ventana nueva...")
plt.show()
print("\n================ PROCESO TERMINADO ================")