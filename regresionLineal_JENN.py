import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ignorar advertencias visuales
warnings.filterwarnings('ignore')

# 1. Configuración de carpetas
script_dir = Path.cwd()
output_folder = script_dir / 'resultados_regresion_lineal'
output_folder.mkdir(exist_ok=True)

plt.style.use('ggplot')

print("=" * 60)
print("EJECUCIÓN DE REGRESIÓN LINEAL: FLUJO PROFESIONAL")
print("=" * 60)

# 2. Generación de Datos de Ejemplo (Ej. Horas de estudio vs Calificación)
print("\n[1/4] Generando datos...")
np.random.seed(42)
# Generamos 100 datos: X = horas (de 0 a 10), y = calificación (con algo de ruido aleatorio)
X = 10 * np.random.rand(100, 1)
y = 5 + 8 * X + np.random.randn(100, 1) * 10 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Datos listos. Entrenamiento: {len(X_train)} muestras. Prueba: {len(X_test)} muestras.")

# 3. Entrenamiento del Modelo
print("\n[2/4] Entrenando modelo de Regresión Lineal...")
modelo = LinearRegression()
modelo.fit(X_train, y_train)

print(f"✓ Ecuación encontrada: Y = {modelo.coef_[0][0]:.2f}X + {modelo.intercept_[0]:.2f}")

# 4. Evaluación del Modelo
print("\n[3/4] Evaluando el modelo...")
predicciones = modelo.predict(X_test)
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f"✓ Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"✓ Coeficiente de Determinación (R2): {r2:.2f} (Entre más cerca de 1.0, mejor)")

# 5. Visualización Avanzada y Guardado
print("\n[4/4] Generando gráfica de la línea de tendencia...")
fig, ax = plt.subplots(figsize=(10, 6))

# Puntos reales
ax.scatter(X_test, y_test, color='#4ECDC4', label='Datos de Prueba (Reales)', s=80, edgecolor='black', alpha=0.7)

# Línea de predicción (Regresión)
ax.plot(X_test, predicciones, color='#FF6B6B', label='Línea de Regresión (Predicción)', linewidth=3)

# Configuración estética
ax.set_title(f'Regresión Lineal Simple\nR² Score: {r2:.4f}', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Variable Independiente (Ej. Horas)', fontsize=12)
ax.set_ylabel('Variable Dependiente (Ej. Resultado)', fontsize=12)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

ruta_imagen = output_folder / 'regresion_lineal_optima.png'
plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')

print(f"✓ Gráfica guardada exitosamente en: {ruta_imagen.absolute()}")

# Mostrar la gráfica en una ventana interactiva
print("✓ Abriendo ventana con la gráfica...")
plt.show()
print("\n================ PROCESO TERMINADO ================")