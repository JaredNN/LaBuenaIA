import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Apagar advertencias de TensorFlow para mantener la consola limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

print("=" * 60)
print("EJECUCIÓN DE RED NEURONAL PROFUNDA (TENSORFLOW/KERAS)")
print("=" * 60)

# 1. Generación de Datos Complejos (Círculos concéntricos)
print("\n[1/3] Generando dataset no lineal (Círculos)...")
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Datos listos. Entrenamiento: {len(X_train)} muestras. Prueba: {len(X_test)} muestras.")

# 2. Construcción de la Arquitectura de la Red Neuronal
print("\n[2/3] Construyendo y compilando la Red Neuronal...")
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu', name="Capa_Oculta_1"),
    tf.keras.layers.Dense(8, activation='relu', name="Capa_Oculta_2"),
    tf.keras.layers.Dense(1, activation='sigmoid', name="Capa_Salida")
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. Entrenamiento del Modelo
print("\n[3/3] Entrenando la Red (100 Épocas)...")
historial = modelo.fit(
    X_train, y_train, 
    epochs=100, 
    validation_split=0.2, 
    verbose=0 
)
print("✓ Entrenamiento completado.")

# Evaluación final en consola
loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
print(f"✓ Accuracy en datos de prueba: {accuracy * 100:.2f}%")

# 4. Visualización en Ventana Emergente
print("\nLanzando gráficas en ventana emergente... (Cierra la ventana para terminar el script)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica de Pérdida (Error)
ax1.plot(historial.history['loss'], label='Error Entrenamiento', color='#FF6B6B', linewidth=2)
ax1.plot(historial.history['val_loss'], label='Error Validación', color='#4ECDC4', linewidth=2)
ax1.set_title('Curva de Pérdida (Loss)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Error (Binary Crossentropy)')
ax1.legend()

# Gráfica de Precisión (Accuracy)
ax2.plot(historial.history['accuracy'], label='Precisión Entrenamiento', color='#FF6B6B', linewidth=2)
ax2.plot(historial.history['val_accuracy'], label='Precisión Validación', color='#4ECDC4', linewidth=2)
ax2.set_title('Curva de Precisión (Accuracy)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Precisión')
ax2.legend()

plt.tight_layout()
plt.show() # <-- Este comando es el que levanta la ventana interactiva

