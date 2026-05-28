import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Descargar y preparar el conjunto de datos CIFAR10
# El conjunto contiene 60,000 imágenes en color en 10 clases.
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar los valores de píxeles para que estén entre 0 y 1.
train_images, test_images = train_images / 255.0, test_images / 255.0

# 2. Verificar los datos
# Trazamos las primeras 25 imágenes del conjunto de entrenamiento.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # Las etiquetas CIFAR son arrays, por eso necesitamos el índice extra.
    plt.xlabel(class_names[train_labels[i][0]])
plt.show(block=False)
plt.pause(2) # Mostramos la gráfica por 2 segundos y continuamos

# 3. Crear la base convolucional
# Definimos una pila de capas Conv2D y MaxPooling2D.
# La entrada es de forma (32, 32, 3).
model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Mostrar la arquitectura base
print("--- Resumen de la Base Convolucional ---")
model.summary()

# 4. Agregar capas densas en la parte superior
# Aplanamos la salida 3D a 1D y agregamos capas densas.
# CIFAR tiene 10 clases de salida, por lo que la capa final tiene 10 salidas.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Mostrar la arquitectura completa
print("\n--- Resumen del Modelo Completo ---")
model.summary()

# 5. Compilar y entrenar el modelo
# Usamos el optimizador 'adam' y SparseCategoricalCrossentropy.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenamos por 10 épocas validando con el conjunto de prueba.
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# 6. Evaluar el modelo
# Graficamos la precisión (accuracy) vs la precisión de validación (val_accuracy).
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluamos el rendimiento final en el conjunto de prueba e imprimimos el resultado.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"\nPrecisión final en prueba: {test_acc}")