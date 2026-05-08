# 📚 Teoría Completa: Data Science y Machine Learning
## De Fundamentos a Modelos Predictivos

---

## 📋 Tabla de Contenidos
1. [NumPy - La Base Numérica](#numpy)
2. [Pandas - Manipulación de Datos](#pandas)
3. [Machine Learning - Conceptos](#ml-conceptos)
4. [Regresión Lineal - Teoría y Práctica](#regresion-lineal)
5. [Random Forest - Ensembles](#random-forest)
6. [Métricas de Evaluación](#metricas)
7. [Feature Engineering](#feature-engineering)
8. [Flujo Completo - Caso Práctico](#flujo-completo)

---

## NUMPY - LA BASE NUMÉRICA {#numpy}

### ¿Por qué NumPy?
Python es lento para cálculos numéricos porque trabaja con listas dinámicas. NumPy usa arreglos estáticos en memoria, optimizados en C, haciéndolos 10-100x más rápidos.

### 1. Arreglos (Arrays)
Un arreglo es una colección de números organizados en dimensiones.

```python
import numpy as np

# Array 1D (vector)
arr = np.array([1, 2, 3, 4, 5])
# Visualización mental: [1, 2, 3, 4, 5]

# Array 2D (matriz)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# Visualización:
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]

# Array 3D (tensor)
tensor = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
```

**Ventaja:** Todos los elementos son del MISMO tipo (int, float), por eso NumPy puede optimizar memoria.

### 2. Creación de Arreglos - Métodos Comunes

#### np.arange() - Rango de valores
```python
np.arange(start, stop, step)

# Ejemplo: 10 a 50
arr = np.arange(10, 51)
# Resultado: [10, 11, 12, ..., 50]

# Con salto de 2
arr = np.arange(0, 10, 2)
# Resultado: [0, 2, 4, 6, 8]
```

#### np.zeros() y np.ones() - Llenar con valores
```python
zeros = np.zeros((3, 4))  # Matriz 3x4 llena de 0s
# [[0., 0., 0., 0.],
#  [0., 0., 0., 0.],
#  [0., 0., 0., 0.]]

ones = np.ones((2, 3))  # Matriz 2x3 llena de 1s
# [[1., 1., 1.],
#  [1., 1., 1.]]
```

#### np.random.randint() - Números aleatorios enteros
```python
# Números aleatorios entre 1 y 20, tamaño (4,5)
arr = np.random.randint(1, 20, size=(4, 5))
```

### 3. Indexación y Slicing

**Indexación 1D:**
```python
arr = np.array([10, 20, 30, 40, 50])
arr[0]      # → 10 (primer elemento)
arr[2]      # → 30 (tercer elemento)
arr[-1]     # → 50 (último elemento)
```

**Slicing 1D (rango):**
```python
arr[1:4]    # → [20, 30, 40] (índices 1,2,3 - NO incluye el 4)
arr[::2]    # → [10, 30, 50] (cada 2 elementos)
arr[:3]     # → [10, 20, 30] (primeros 3)
```

**Indexación 2D:**
```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
matrix[0]       # → [1, 2, 3] (primera fila)
matrix[1, 2]    # → 6 (fila 1, columna 2)
matrix[0, :]    # → [1, 2, 3] (primera fila, todas las columnas)
matrix[:, 0]    # → [1, 4, 7] (todas las filas, primera columna)
```

### 4. Reshape - Cambiar Dimensiones

Reshape NO cambia los datos, solo su estructura:

```python
arr = np.arange(1, 13)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Convertir a matriz 3x4 (3 filas, 4 columnas)
matrix = arr.reshape(3, 4)
# [[1,  2,  3,  4],
#  [5,  6,  7,  8],
#  [9, 10, 11, 12]]

# Convertir a matriz 2x6
matrix2 = arr.reshape(2, 6)
# [[1,  2,  3,  4,  5,  6],
#  [7,  8,  9, 10, 11, 12]]

# Regla: los elementos totales deben ser iguales
# 12 elementos → 3x4, 2x6, 12x1, etc. TODO funciona
# Pero 12 → 3x5 NO funciona (15 ≠ 12)
```

### 5. Operaciones Elemento a Elemento

Las operaciones en NumPy se aplican a CADA elemento:

```python
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

# Suma
x + y  # → [6, 8, 10, 12]

# Multiplicación
x * y  # → [5, 12, 21, 32]

# Con escalares
x + 10  # → [11, 12, 13, 14]
x * 2   # → [2, 4, 6, 8]
```

**Esto es CRUCIAL para data science:** cuando trabajas con 1 millón de números, NumPy los procesa todos en paralelo en C, no en Python lento.

### 6. Funciones Universales (Ufuncs)

```python
arr = np.array([1, 4, 9, 16])

np.sqrt(arr)   # → [1, 2, 3, 4] (raíz cuadrada)
np.log(arr)    # → logaritmo natural
np.exp(arr)    # → e^x para cada elemento
np.sin(arr)    # → seno trigonométrico
np.abs(arr)    # → valor absoluto
```

---

## PANDAS - MANIPULACIÓN DE DATOS {#pandas}

### ¿Por qué Pandas después de NumPy?
NumPy es para arrays numéricos puros. Pandas es para datos REALES:
- Tablas con nombres de columnas
- Datos faltantes
- Tipos mixtos (números, texto, fechas)
- Operaciones tipo SQL

### 1. Series - Arrays Etiquetados

Una Serie es un array 1D con etiquetas (índices):

```python
import pandas as pd

# Sin etiquetas (se asignan 0, 1, 2...)
s = pd.Series([10, 20, 30])
# 0    10
# 1    20
# 2    30
# dtype: int64

# Con etiquetas personalizadas
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
# a    10
# b    20
# c    30

# Acceder
s['a']      # → 10
s[0]        # → 10 (también funciona número)
```

### 2. DataFrame - Tablas Reales

Un DataFrame es varias Series alineadas por índice (como tabla SQL):

```python
data = {
    'Nombre': ['Alice', 'Bob', 'Charlie'],
    'Edad': [24, 30, 22],
    'Salario': [50000, 60000, 45000]
}

df = pd.DataFrame(data)
#     Nombre  Edad  Salario
# 0    Alice    24    50000
# 1      Bob    30    60000
# 2  Charlie    22    45000

# Acceder a columna
df['Nombre']  # → Serie con los nombres

# Acceder a fila
df.loc[0]     # → Primera fila como Serie

# Acceder a celda
df.loc[0, 'Nombre']  # → 'Alice'
df.iloc[0, 1]        # → 24 (por posición numérica)
```

### 3. Crear DataFrames desde Diferentes Fuentes

**Desde diccionario:**
```python
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
```

**Desde lista de diccionarios:**
```python
data = [
    {'Name': 'Alice', 'Age': 25},
    {'Name': 'Bob', 'Age': 30}
]
df = pd.DataFrame(data)
```

**Desde CSV (IMPORTANTE para ML):**
```python
df = pd.read_csv('datos.csv')
```

### 4. Operaciones Clave en DataFrames

#### Inspección
```python
df.head()        # Primeras 5 filas
df.tail()        # Últimas 5 filas
df.info()        # Tipos, no-nulos
df.describe()    # Estadísticas: media, std, min, max
df.shape         # (filas, columnas)
```

#### Limpieza
```python
df.dropna()           # Eliminar filas con faltantes
df.fillna(valor)      # Rellenar con valor
df['col'].astype(int) # Convertir tipo
```

#### Agregar/Eliminar Columnas
```python
df['nueva_col'] = [1, 2, 3]           # Agregar columna
df['descuento'] = df['precio'] * 0.1  # Con operación
df.drop('col', axis=1)                # Eliminar columna
```

#### Filtrar Filas
```python
df[df['Edad'] > 25]           # Filas donde Edad > 25
df[(df['Edad'] > 25) & (df['Salario'] > 50000)]  # Múltiples condiciones
```

#### Ordenar
```python
df.sort_values('Edad')                # Ascendente
df.sort_values('Edad', ascending=False)  # Descendente
```

#### Funciones Agregadas
```python
df['Edad'].mean()    # Promedio
df['Edad'].std()     # Desviación estándar
df['Salario'].sum()  # Suma
df['Salario'].max()  # Máximo
df.groupby('Departamento')['Salario'].mean()  # Promedio por grupo
```

---

## MACHINE LEARNING - CONCEPTOS {#ml-conceptos}

### ¿Qué es Machine Learning?

Hacer que una máquina aprenda patrones de datos para hacer predicciones sin ser programada explícitamente.

**Analogía:** En vez de escribir reglas como "si motor > 200hp y peso > 3000kg entonces precio = X", le muestras miles de ejemplos y deja que el modelo DESCUBRA las reglas.

### Tipos de ML

#### 1. Supervisado - Tiene etiqueta (y)
Tienes pares entrada-salida. Aprendes a predecir la salida dada una entrada.

**Regresión:** Predecir número continuo
- Ejemplo: predecir PRECIO de auto (5000, 15234.50, 45000)
- Salida: número real

**Clasificación:** Predecir categoría
- Ejemplo: predecir si AUTO ES DEPORTIVO (Sí/No) o MARCA (Honda, Toyota, BMW)
- Salida: categoría

#### 2. No supervisado
Sin etiqueta. Buscas patrones ocultos.
- Clustering: agrupar datos similares
- Dimensionalidad: reducir variables

#### 3. Refuerzo
Aprende por recompensas (videojuegos, robots).

### Componentes del ML Supervisado

```
DATOS
  ↓
[FEATURES (X)] → [MODELO] → [PREDICCIÓN (ŷ)]
[TARGET (y)]
```

**Features (X):** Variables de entrada
- Ejemplo: motor, potencia, mpg, peso

**Target (y):** Lo que queremos predecir
- Ejemplo: precio

**Modelo:** Función que aprende relación X → y
- Regresión lineal, Random Forest, Neural Networks, etc.

### Flujo de Trabajo ML

```
1. Cargar datos
      ↓
2. Explorar y limpiar
      ↓
3. Seleccionar features
      ↓
4. Dividir train/test (80/20)
      ↓
5. Entrenar modelo
      ↓
6. Predecir en test
      ↓
7. Evaluar métricas
      ↓
8. ¿Bueno? → Usar modelo
   ¿Malo? → Mejorar features/modelo
```

---

## REGRESIÓN LINEAL - TEORÍA Y PRÁCTICA {#regresion-lineal}

### Idea Básica

Encontrar la línea recta que mejor ajuste tus datos.

**En 1D (una variable):**
```
y = m*x + b

Donde:
- y es la predicción (precio)
- x es la entrada (horsepower)
- m es la pendiente (cuánto cambia y por cada x)
- b es el intercepto (valor de y cuando x=0)
```

**Visualización:**
```
Precio ($)
   |     ●
   |    ● ●
   |   ●   ●
   |  ●     ●  ← Esta recta es la que queremos encontrar
   | ●       ●
   |●---------●
   +--------------- Horsepower
```

El objetivo: encontrar m y b que minimicen la distancia promedio entre la línea y los puntos.

### En Múltiples Variables

**Regresión Lineal Múltiple:**
```
y = b₀ + b₁*x₁ + b₂*x₂ + ... + bₙ*xₙ

Ejemplo con autos:
Precio = b₀ + b₁*motor + b₂*potencia + b₃*mpg
```

Es la misma idea, pero con más dimensiones (imposible de visualizar en 3D+).

### ¿Cómo Aprende el Modelo?

**Minimización del Error - Mínimos Cuadrados:**

1. Haces una predicción: ŷ = m*x + b
2. Calculas el error de cada punto: error = y - ŷ
3. Para evitar que errores positivos y negativos se cancelen, los elevas al cuadrado
4. Promedias todos los errores al cuadrado
5. Encuentras m y b que minimizan este promedio

Matemáticamente:
```
Minimizar: MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

**¿Por qué al cuadrado?**
- Errores grandes se penalizan MÁS (50² = 2500 vs 10² = 100)
- Siempre positivo (no se cancelan)
- Matemáticamente elegante para encontrar mínimo

### Código en Práctica

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Datos
X = df[['horsepower']]      # Features (matriz 2D)
y = df['price']             # Target (vector 1D)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# test_size=0.2 → 80% entrenamiento, 20% prueba
# random_state=42 → misma división cada vez (reproducibilidad)

# Crear y entrenar
model = LinearRegression()
model.fit(X_train, y_train)

# Parámetros aprendidos
print(model.coef_)       # La pendiente m
print(model.intercept_)  # El intercepto b

# Predicción
y_pred = model.predict(X_test)
```

### Ventajas y Desventajas

**Ventajas:**
✓ Rápido de entrenar
✓ Fácil de interpretar (coeficientes = importancia)
✓ Buen punto de partida

**Desventajas:**
✗ Asume relación LINEAL (línea recta)
✗ Si datos son curvos, funciona mal
✗ Sensible a outliers (valores atípicos)

---

## RANDOM FOREST - ENSEMBLES {#random-forest}

### Problema de la Regresión Lineal

Si tus datos NO son lineales (tienen curvas), la línea recta no es suficiente:

```
Datos reales (curvos)     Regresión lineal fallando
       ●                         |/
      ● ●                       / |
     ●   ●                     /  ●
    ●     ●                   /  ● ●
   ●       ●   vs   --------/-- ● ● ●
  ●         ●              /    ●   ●
 ●           ●            /    ●     ●
            ← No es recto!
```

### Idea de Random Forest

En vez de una línea recta, crear un "bosque" de árboles de decisión.

**Árbol de Decisión - Ejemplo Simplificado:**
```
                    Motor > 100?
                   /            \
                Sí/              \No
               /                  \
        Potencia > 80?        Potencia > 50?
        /        \            /        \
      Sí/        \No        Sí/        \No
     /            \        /            \
  $35000      $18000   $22000      $12000
```

Cada decisión divide el espacio en rectángulos, los datos en cada rectángulo usan el mismo precio predicho.

**Ventaja sobre la línea recta:**
```
Árbol de decisión (rectángulos)   vs   Regresión lineal (línea)
      ┌─────┐
      │     │
   ┌──┤     ├──┐
   │  │     │  │
   └──┤     ├──┘
      │     │
      └─────┘
Puede aprender formas complejas    Solo aprender líneas
```

### ¿Por qué RANDOM Forest?

Un solo árbol puede SOBREAJUSTARSE (memorizar ruido).

**Random Forest = Múltiples árboles + Promedio:**

1. Creas 100 árboles (o más)
2. Cada árbol ve datos DIFERENTES (muestreo aleatorio)
3. Cada árbol hace una predicción
4. Promedias todas las predicciones

**Ejemplo:**
```
Árbol 1 predice: $25000
Árbol 2 predice: $26000
Árbol 3 predice: $24500
...
Promedio: $25166  ← Predicción final más robusta
```

### Hiperparámetros Principales

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,           # Número de árboles (más = mejor pero lento)
    max_depth=10,               # Profundidad máxima (más profundo = más complejo)
    min_samples_split=5,        # Mínimo de muestras para dividir (previene overfitting)
    min_samples_leaf=2,         # Mínimo en hoja final
    random_state=42             # Para reproducibilidad
)

model.fit(X_train, y_train)
```

**Ajuste de Hiperparámetros:**
```
max_depth pequeño  → Modelo simple, puede underfittear
max_depth grande   → Modelo complejo, puede overfittear

Es un equilibrio: No tan simple que no aprenda, 
                  no tan complejo que memorice ruido
```

### Importancia de Variables

Random Forest te dice cuáles features son MÁS importantes:

```python
feature_importance = model.feature_importances_
# [0.15, 0.45, 0.25, 0.15]  ← Importancia de cada variable

# Motor es 45% importante
# Peso es 25% importante
# etc.
```

---

## MÉTRICAS DE EVALUACIÓN {#metricas}

### El Problema del Train vs Test

Si evalúas SOLO en los datos de entrenamiento, el modelo puede estar "memorizando":

```
Train Accuracy: 95%  ← Vio estos datos
Test Accuracy: 60%   ← Datos nuevos, no vió
```

SIEMPRE evalúa en TEST.

### Métricas para Regresión

#### 1. Mean Squared Error (MSE)

```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

Interpretación:
- Promedio de errores al cuadrado
- Penaliza errores grandes MÁS
- Unidades: cuadrado de la variable (precio²)
- Más bajo = mejor
- No es interpretable en términos reales

Ejemplo: MSE = 5,000,000
¿Qué significa? Difícil decir...
```

#### 2. Root Mean Squared Error (RMSE) - MÁS INTERPRETABLE

```
RMSE = √MSE

Ventaja: Mismas unidades que y (dinero)

Ejemplo: RMSE = $2,361
Interpretación clara: "En promedio, me equivoco por $2,361"
```

**Cálculo paso a paso:**
```
Reales:       [10000, 15000, 20000]
Predichos:    [9500,  16000, 19500]
Errores:      [500,   -1000, 500]
Errores²:     [250000, 1000000, 250000]
Promedio:     1,500,000 / 3 = 500,000
RMSE:         √500,000 ≈ $707
```

#### 3. R² Score (Coeficiente de Determinación)

```
R² = 1 - (SS_res / SS_tot)

Donde:
- SS_res = Σ(yᵢ - ŷᵢ)²  (suma de errores residuales)
- SS_tot = Σ(yᵢ - ȳ)²   (suma total de variación)
```

**Interpretación:**
```
R² = 0.87 → El modelo explica 87% de la variabilidad
            Muy bueno

R² = 0.50 → El modelo explica 50% de la variabilidad
            Moderado

R² = 0.10 → El modelo explica 10% de la variabilidad
            Malo
```

**Visualización:**
```
Si predijeras siempre la media:     0% (R² ≈ 0)
Si predijeras perfectamente:        100% (R² = 1)
Tu modelo está en el medio
```

### Código Completo de Evaluación

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predicción
y_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: ${mse:,.0f}")
print(f"RMSE: ${rmse:,.0f}")
print(f"R² Score: {r2:.4f}")

# Ejemplo de output real:
# MSE: $5,000,000
# RMSE: $2,236
# R² Score: 0.6870
# Interpretación: Me equivoco en promedio por $2,236, 
#                 y explico 68.7% de la varianza
```

---

## FEATURE ENGINEERING {#feature-engineering}

### ¿Por qué Feature Engineering?

Los datos crudos NO siempre contienen la información que necesitas.

**Ejemplo:** Quieres predecir precio, tienes motor y peso. Pero...

Un auto con motor pequeño pero ultraligero podría ser RÁPIDO (poder/peso alto = rápido = caro).

Así que CREAS la variable: `power_to_weight = horsepower / weight`

### Técnicas Comunes

#### 1. Variables Derivadas

```python
# Power to size ratio
df['power_to_size'] = df['horsepower'] / df['engine-size']

# Fuel efficiency average
df['avg_efficiency'] = (df['city-mpg'] + df['highway-mpg']) / 2

# Age of dataset (si tienes year)
df['age'] = 2024 - df['year']
```

#### 2. Polinomios

Si la relación es curvada, agrega el cuadrado:

```python
# Relación curvada: precio sube más con motor grande
df['horsepower_squared'] = df['horsepower'] ** 2

# Modelo polinómico (en regresión lineal):
y = b₀ + b₁*x + b₂*x²
```

#### 3. Codificación de Categóricas (One-Hot Encoding)

Las máquinas aprenden números, no texto:

```
Columna: marca
Ford
Honda
BMW

Se convierte en:
marca_Ford  marca_Honda  marca_BMW
    1            0           0      ← Ford
    0            1           0      ← Honda
    0            0           1      ← BMW
```

**En Pandas:**
```python
df_encoded = pd.get_dummies(df, columns=['marca'], drop_first=True)
```

#### 4. Escalado (Standardization/Normalization)

Poner todas las variables en la misma escala:

```
Motor: 61-326 (rango 265)
Peso: 1488-4066 (rango 2578)
Precio: 5118-45400 (rango 40282)

Un cambio de 1 en motor vs 1 en peso son DIFERENTES en importancia.
Muchos modelos funcionan mejor si todas están 0-1 o media 0 desv 1.

StandardScaler:
1. Resta la media
2. Divide entre desviación estándar
```

---

## CLASIFICACIÓN EN MACHINE LEARNING {#clasificacion}

### Diferencia: Regresión vs Clasificación

**REGRESIÓN:** Predecir número CONTINUO
```
Entrada: motor, potencia, mpg
Salida: PRECIO ($25000, $45234.50, $60000)
Ejemplo: Precio de auto
```

**CLASIFICACIÓN:** Predecir CATEGORÍA
```
Entrada: motor, potencia, mpg, edad, sexo
Salida: SUPERVIVENCIA (Sí/No), DEPORTIVO (Sí/No)
Ejemplo: ¿Sobrevivió en Titanic? (0 o 1)
```

### Casos de Uso de Clasificación

- Diagnóstico médico (enfermo/sano)
- Spam detection (spam/no spam)
- Credit approval (aprobado/rechazado)
- Titanic survival (sobrevivió/no sobrevivió)

### Modelos de Clasificación

#### 1. Logistic Regression (Regresión Logística)

A pesar del nombre, es para CLASIFICACIÓN, no regresión.

**Idea:**
- Usa una función sigmoid que convierte números en probabilidades (0-1)
- Output: probabilidad de pertenencia a la clase 1
- Si probabilidad > 0.5 → Clase 1 (Sobrevivió)
- Si probabilidad ≤ 0.5 → Clase 0 (No sobrevivió)

**Fórmula:**
```
P(Clase=1) = 1 / (1 + e^(-z))

Donde z = b₀ + b₁*x₁ + b₂*x₂ + ... + bₙ*xₙ
```

**Ventajas:**
✓ Rápida de entrenar
✓ Interpretable (coeficientes = importancia)
✓ Excelente para relaciones lineales
✓ Requiere menos datos

**Desventajas:**
✗ Asume relación lineal entre features y probabilidad
✗ No captura interacciones complejas
✗ Sensible a outliers

**Código:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # [[prob_0, prob_1], ...]
```

#### 2. Random Forest Classifier

Igual que RandomForestRegressor, pero para clasificación.

**Diferencia clave:**
- En regresión: promedias las predicciones numéricas
- En clasificación: cada árbol "vota" por una clase, tomas la moda (clase más votada)

**Ejemplo:**
```
Árbol 1 predice: Sobrevivió (1)
Árbol 2 predice: No sobrevivió (0)
Árbol 3 predice: Sobrevivió (1)
...
Votación final: 2 votos por "Sobrevivió" → Predicción: Sobrevivió
```

**Ventajas:**
✓ Captura relaciones NO lineales
✓ Maneja valores faltantes bien
✓ Importancia automática de features
✓ Robusto a outliers

**Desventajas:**
✗ Más lento que Logistic Regression
✗ Riesgo de overfitting (ajustarse demasiado)
✗ Menos interpretable

**Código:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # [[prob_0, prob_1], ...]
```

---

## MÉTRICAS DE CLASIFICACIÓN {#metricas-clasificacion}

A diferencia de regresión, las métricas son diferentes.

### Confusion Matrix (Matriz de Confusión)

```
                    Predicción
                  Neg    Pos
Real  Neg  │  TN  │  FP  │
           ├──────┼──────┤
      Pos  │  FN  │  TP  │
           └──────┴──────┘

TN (True Negative):  Predijo No Sobrevivió Y no sobrevivió ✓
FP (False Positive): Predijo Sobrevivió PERO no sobrevivió ✗
FN (False Negative): Predijo No Sobrevivió PERO sobrevivió ✗
TP (True Positive):  Predijo Sobrevivió Y sobrevivió ✓
```

### Métrica 1: Accuracy (Exactitud)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Interpretación: % de predicciones correctas
Rango: 0 a 1 (mejor = 1)

Ejemplo:
TP=70, TN=50, FP=10, FN=20
Accuracy = (70+50)/(70+50+10+20) = 120/150 = 0.80 = 80%
```

**PROBLEMA:** Si tienes clase desbalanceada (90% sobrevivió, 10% no)
- Modelo "tonto" que predice siempre "Sobrevivió" = 90% accuracy!
- Pero FALLA completamente en predecir "No sobrevivió"

### Métrica 2: Precision (Precisión)

```
Precision = TP / (TP + FP)

Interpretación: De los que predije "Sobrevivió", ¿cuántos realmente sobrevivieron?
Cuando es importante NO tener falsos positivos

Ejemplo: 
Email spam detection
- NO quieres marcar emails legítimos como spam
- Precision alta = menos emails legítimos perdidos
```

### Métrica 3: Recall (Sensibilidad)

```
Recall = TP / (TP + FN)

Interpretación: De los que realmente sobrevivieron, ¿cuántos predije?
Cuando es importante NO tener falsos negativos

Ejemplo:
Detección de cáncer
- NO quieres perder a pacientes enfermos
- Recall alto = menos enfermos sin tratamiento
```

### Métrica 4: F1-Score (Media Armónica)

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Interpretación: Promedio balanceado de precision y recall
Usa cuando quieres balance entre ambas

Rango: 0 a 1 (mejor = 1)

Ejemplo:
Precision=0.8, Recall=0.6
F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.96 / 1.4 ≈ 0.686
```

### Métrica 5: ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

```
Interpretación: Medida de capacidad discriminativa del modelo
Qué tan bien el modelo separa las dos clases

ROC = 0.5: Modelo tan bueno como random guessing
ROC = 1.0: Modelo perfecto
ROC < 0.5: Modelo peor que random (invierte predicciones)
```

### Código de Métricas

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad clase 1

# Todas las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
# Output: [[TN, FP],
#          [FN, TP]]

# Reporte automático
print(classification_report(y_test, y_pred, target_names=['No Sobrevivió', 'Sobrevivió']))
```

---

## FLUJO COMPLETO - COMPARACIÓN DE MODELOS: TITANIC {#flujo-titanic}

### Problema: Predecir Supervivencia en el Titanic

Clasificación Binaria: Sobrevivió (1) o No (0)

#### Paso 1: Cargar y Explorar

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Cargar
df = pd.read_csv('Titanic.csv')

# Explorar
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

#### Paso 2: Limpiar Datos

```python
# Rellenar valores faltantes
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Eliminar columnas innecesarias
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Codificar variables categóricas
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

#### Paso 3: Preparar Features y Target

```python
# Feature engineering (opcional)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Features y target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y asegura que train/test tengan la misma proporción de clases

# Escalar (importante para Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Paso 4: Entrenar Modelo 1 - Logistic Regression

```python
# Crear y entrenar
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predicción
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluar
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_roc_auc = roc_auc_score(y_test, y_proba_lr)

print("LOGISTIC REGRESSION")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"F1-Score: {lr_f1:.4f}")
print(f"ROC-AUC: {lr_roc_auc:.4f}")
print(classification_report(y_test, y_pred_lr))
```

#### Paso 5: Entrenar Modelo 2 - Random Forest

```python
# Crear y entrenar (NO necesita escalado)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predicción
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluar
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_proba_rf)

print("RANDOM FOREST")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"F1-Score: {rf_f1:.4f}")
print(f"ROC-AUC: {rf_roc_auc:.4f}")
print(classification_report(y_test, y_pred_rf))
```

#### Paso 6: Comparar y Elegir el Mejor

```python
# Comparación
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [lr_accuracy, lr_f1, lr_roc_auc],
    'Random Forest': [rf_accuracy, rf_f1, rf_roc_auc]
})

print(comparison)

# Elegir mejor
best_model = 'Random Forest' if rf_accuracy > lr_accuracy else 'Logistic Regression'
print(f"\nBest Model: {best_model}")
```

#### Paso 7: Análisis con el Mejor Modelo

```python
# Si es Random Forest, ver importancia de features
if rf_accuracy > lr_accuracy:
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance:")
    print(importance)

# Analizar errores
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()

print(f"\nSobrevivientes predichos correctamente: {tp} de {tp+fn}")
print(f"No-sobrevivientes predichos correctamente: {tn} de {tn+fp}")
```

### Conclusiones del Análisis

**Random Forest típicamente mejor porque:**
1. Captura relaciones NO lineales en los datos
2. Algunas características interactúan (edad + sexo = supervivencia)
3. Más robusto a outliers (hay pasajeros atípicos)

**Logistic Regression mejor en casos:**
- Datos claramente separables linealmente
- Necesitas modelo muy interpretable
- Tienes muy pocos datos

---

## FLUJO COMPLETO - CASO PRÁCTICO {#flujo-completo}

### Predicción de Precio de Autos - De Cero

#### Paso 1: Cargar y Explorar

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar
df = pd.read_csv('Automobile_price_data.csv')

# Explorar
print(df.head())        # Primeras filas
print(df.info())        # Tipos y no-nulos
print(df.describe())    # Estadísticas
print(df.isnull().sum()) # Conteo de faltantes
```

#### Paso 2: Limpiar Datos

```python
# Reemplazar valores faltantes
df = df.replace('?', np.nan)          # Marcar como NaN
df = df.dropna()                       # Eliminar filas con NaN

# Convertir a tipo correcto
df['price'] = pd.to_numeric(df['price'])
df['horsepower'] = pd.to_numeric(df['horsepower'])
df['engine-size'] = pd.to_numeric(df['engine-size'])
```

#### Paso 3: Feature Engineering

```python
# Variables derivadas
df['power_to_size'] = df['horsepower'] / df['engine-size']
df['avg_mpg'] = (df['city-mpg'] + df['highway-mpg']) / 2

# Codificar categóricas
df_encoded = pd.get_dummies(df, 
                            columns=['make', 'fuel-type', 'body-style'],
                            drop_first=True)
```

#### Paso 4: Preparar X y y

```python
# Seleccionar features
features = ['engine-size', 'horsepower', 'city-mpg', 'highway-mpg',
            'power_to_size', 'avg_mpg', 'curb-weight']
X = df_encoded[features]
y = df_encoded['price']

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% entrena, 20% prueba
```

#### Paso 5: Entrenar Modelo

```python
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)
# Aquí el modelo APRENDE relaciones en X_train → y_train
```

#### Paso 6: Predecir y Evaluar

```python
# Predicción en test
y_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Comparar train vs test para detectar overfitting
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)

print(f"R² Train: {r2_train:.4f}")
print(f"R² Test: {r2_test:.4f}")

if r2_train - r2_test > 0.1:
    print("⚠️ Posible overfitting - modelo se ajustó demasiado a train")
```

#### Paso 7: Mejorar Modelo (Tuning)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Definir espacio de búsqueda
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 10, 15, 20, 25],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
}

# RandomizedSearchCV prueba combinaciones aleatorias
random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,          # Prueba 20 combinaciones
    cv=5,               # 5-fold cross-validation
    random_state=42
)

random_search.fit(X_train, y_train)

# Mejores parámetros encontrados
print("Mejores parámetros:", random_search.best_params_)

# Evaluar con mejor modelo
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)

print(f"R² mejorado: {r2_best:.4f}")
```

#### Paso 8: Análisis de Importancia

```python
# ¿Cuáles features son más importantes?
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Visualizar
import matplotlib.pyplot as plt
plt.barh(feature_importance['Feature'][:10], 
         feature_importance['Importance'][:10])
plt.xlabel('Importancia')
plt.title('Top 10 Features')
plt.show()
```

### Checklist de Decisiones

```
DURANTE EL PROYECTO, PREGÚNTATE:

1. ¿Los datos están limpios?
   → Revisé faltantes, outliers, tipos
   
2. ¿Qué variables creo que predirán bien?
   → Exploro correlaciones
   
3. ¿Necesito crear nuevas variables?
   → Feature engineering
   
4. ¿Qué modelo usar?
   → Empiezo simple (regresión lineal)
   → Pruebo complex (Random Forest)
   
5. ¿El modelo generaliza?
   → Comparo train vs test
   → Si difieren mucho → overfitting
   
6. ¿Qué features son importantes?
   → Veo importancia/coeficientes
   → Puedo eliminar las que no aportan
   
7. ¿Cómo puedo mejorar?
   → Mejor limpieza
   → Más features
   → Tuning de hiperparámetros
```

---

## 🎯 Conclusión

Estos son los bloques fundamentales de Machine Learning aplicado:

1. **NumPy:** Arrays eficientes para cálculo numérico
2. **Pandas:** Manipulación de datos tabulares reales
3. **ML Supervisado:** Aprender relaciones X → y
4. **Regresión:** Predecir números continuos
5. **Evaluación:** Métricas para saber si el modelo funciona
6. **Mejora:** Feature engineering y tuning

Ahora que entiendes la TEORÍA, en el siguiente paso aprenderemos a **APLICARLA sistemáticamente a nuevos problemas.**

---

*Documento creado por: Tu asistente de IA*
*Fecha: Mayo 2026*
*Enfoque: Educativo, paso a paso, sin tecnicismos innecesarios*
