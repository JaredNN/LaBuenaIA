import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Iniciando proceso...\nCargando datos...")
data = pd.read_csv('./Dataset/dataset-tortuga.csv')
print(f"Dataset cargado exitosamente: {data.shape[0]} filas, {data.shape[1]} columnas\n")

# Inspección rápida del dataset (Del Código 1)
data.info()
print("\n")

# Selección explícita de variables (Del Código 2 - Más seguro)
features = [
    'HOURS_DATASCIENCE', 'HOURS_BACKEND', 'HOURS_FRONTEND',
    'NUM_COURSES_BEGINNER_DATASCIENCE', 'NUM_COURSES_BEGINNER_BACKEND',
    'NUM_COURSES_BEGINNER_FRONTEND', 'NUM_COURSES_ADVANCED_DATASCIENCE',
    'NUM_COURSES_ADVANCED_BACKEND', 'NUM_COURSES_ADVANCED_FRONTEND',
    'AVG_SCORE_DATASCIENCE', 'AVG_SCORE_BACKEND', 'AVG_SCORE_FRONTEND'
]

# Imputación limpia: solo afecta a las columnas numéricas
print("Imputando valores nulos con la media...")
imputer = SimpleImputer(strategy='mean')
data[features] = pd.DataFrame(imputer.fit_transform(data[features]), columns=features)

# Definición de X e y
X = data[features]
y = data['PROFILE']

# División de datos (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Búsqueda del mejor valor de K
print("\nEvaluando el mejor valor de K (de 3 a 29)...")
acc = {}
for k in range(3, 30, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc[k] = accuracy_score(y_test, y_pred)

# Creación de la gráfica mejorada
plt.figure(figsize=(8, 5))
plt.plot(range(3, 30, 2), list(acc.values()), marker='o', linestyle='-', color='b')
plt.title('Precisión del modelo KNN vs Valor de K')
plt.xlabel('Valor de K (Número de vecinos)')
plt.ylabel('Precisión (Accuracy)')
plt.grid(True)
plt.show()
print("-> Gráfica mostrada en una ventana.")

# Entrenamiento del modelo final con K=13
print("\nEntrenando modelo final con K=13...")
knn_final = KNeighborsClassifier(n_neighbors=13)
knn_final.fit(X_train_scaled, y_train)
y_pred_final = knn_final.predict(X_test_scaled)

# Resultados
print("\n================ RESULTADOS DEL MODELO ================")
print(f"Accuracy Global: {accuracy_score(y_test, y_pred_final):.4f}\n")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_final))