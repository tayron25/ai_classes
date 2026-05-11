# Questions and Answers

## 2026-05-11

### Duda
```python
X = df[['engine-size', 'horsepower', 'city-mpg', 'highway-mpg']]
y = df['price']
```

Que hace cada cosa en esa parte del codigo, por que se selecciona una cosa y no otra, y por que X tiene varias columnas mientras y solo una.

### Resolucion
- En aprendizaje supervisado, `X` son las entradas (features) y `y` es la salida objetivo.
- `X` usa varias columnas porque el modelo se apoya en varios factores para predecir.
- `y` usa una columna porque aqui solo se quiere predecir una variable: `price`.
- `df[[...]]` (doble corchete) devuelve varias columnas en forma de DataFrame.
- `df['...']` (corchete simple) devuelve una sola columna en forma de Serie.
- Se eligieron esas variables por su relacion esperada con el precio (tamano de motor, potencia y consumo).

---

### Duda
¿Por qué muchos árboles en Random Forest es mejor que un solo árbol? ¿Se podría solucionar el sobreajuste de un solo árbol simplemente agregándole más datos? ¿Random Forest es solo para cuando tienes pocos datos?

### Resolucion
- Un solo árbol tiende al sobreajuste: memoriza patrones específicos de los datos, aunque tengas muchos datos.
- Agregar más datos al árbol solo ayuda un poco, pero no resuelve el problema de sobreajuste inherente a un solo árbol.
- Random Forest funciona porque cada árbol ve una **muestra y variables distintas**, lo que crea diversidad.
- Esa diversidad hace que los errores de cada árbol sean **distintos y se cancelen** cuando promedias todos.
- Random Forest es útil tanto con pocos como con muchos datos; no es solo para datos escasos.
- Ejemplo: 1 persona con mucha info sigue cometiendo errores; 100 personas distintas con info parcial dan mejor respuesta.

---

### Duda
¿Qué almacena `regressor`? ¿`fit` entrena? ¿Cómo es ese entrenamiento?

### Resolucion
- `regressor` es el objeto del modelo: al inicio solo tiene la configuración, y después de entrenarlo guarda el aprendizaje.
- `.fit(X_train, y_train)` es el método que entrena el modelo.
- El entrenamiento consiste en mirar las entradas `X_train` y los valores reales `y_train` para aprender patrones.
- En Random Forest, ese proceso construye muchos árboles con muestras distintas de los datos.
- Al final, `regressor` almacena los árboles entrenados y las reglas necesarias para predecir nuevos casos.

## ¿Que es XGBoost? (explicacion a fondo)
XGBoost significa **eXtreme Gradient Boosting**. Es un algoritmo de *ensemble* que combina muchos arboles de decision pequenos para formar un modelo fuerte.

### 1. Idea central: boosting
- En boosting, los arboles se entrenan **uno tras otro**.
- Cada nuevo arbol intenta corregir los errores que cometieron los arboles anteriores.
- Al final, se suman las predicciones de todos los arboles para obtener la decision final.

### 2. Gradient Boosting (de donde sale XGBoost)
- El modelo se entrena minimizando una funcion de perdida (por ejemplo, logloss para clasificacion).
- Se usa el **gradiente** de esa perdida para guiar como debe corregirse el siguiente arbol.
- Por eso se llama *gradient boosting*: mejora el modelo usando el gradiente del error.

### 3. ¿Que hace especial a XGBoost?
XGBoost agrega varias mejoras practicas que lo hacen rapido y preciso:
- **Regularizacion (L1 y L2)**: evita sobreajuste penalizando modelos muy complejos.
- **Submuestreo de filas y columnas**: introduce aleatoriedad para generalizar mejor.
- **Manejo eficiente de datos**: esta optimizado para velocidad y memoria.
- **Soporte para valores faltantes**: aprende automaticamente el mejor camino cuando falta un dato.
- **Paralelizacion**: entrena mas rapido usando varios nucleos.

### 4. ¿Como decide cada arbol?
Cada arbol divide los datos con reglas del tipo:
- si `Age` < 16 entonces...
- si `Sex` = male entonces...
Al sumar muchos arboles, se logra una frontera de decision mas flexible.

### 5. ¿Por que funciona bien en Titanic?
En Titanic hay relaciones fuertes como `Sex`, `Pclass` y `Age`.
XGBoost puede capturar esas relaciones no lineales y combinarlas de forma potente.

### 6. Riesgos y buenas practicas
- Puede **sobreajustar** si se usa mucha profundidad o demasiados arboles.
- Es importante validar con un conjunto de prueba o cross-validation.
- Ajustar hiperparametros suele mejorar el rendimiento.

En resumen, XGBoost es un modelo muy potente porque combina muchos arboles simples, corrigiendo errores paso a paso y usando regularizacion para no memorizar los datos.

---

### Duda
¿Cuál es la diferencia entre accuracy y recall?

### Resolucion
- **Accuracy (Exactitud)**: Es el porcentaje de **todas** tus predicciones que fueron correctas.
  - Fórmula: (Aciertos) / (Total de predicciones)
  - Responde: ¿De todo lo que predije, cuánto acerté?

- **Recall (Sensibilidad)**: Es el porcentaje de casos **positivos reales** que tu modelo encontró correctamente.
  - Fórmula: (Positivos encontrados correctamente) / (Todos los positivos reales)
  - Responde: De los casos positivos reales, ¿cuántos logré detectar?

**Diferencia con ejemplo Titanic:**
- **Accuracy**: De todas las predicciones que hice, ¿cuántas fueron correctas? (Tanto sobrevivientes como no sobrevivientes)
- **Recall**: De todas las personas que **realmente sobrevivieron**, ¿cuántas las predije correctamente?

**¿Cuándo importa cada una?**
- Usa **Accuracy** cuando los errores en ambas clases son igual de graves.
- Usa **Recall** cuando es crítico encontrar todos los casos positivos (ej: detectar enfermedades, fraudes).

---

### Duda
¿Qué es la curva ROC y para qué sirve?

### Resolucion
- La curva ROC muestra cómo cambia el desempeño del modelo al variar el umbral de clasificación.
- El eje X representa la tasa de falsos positivos.
- El eje Y representa la tasa de verdaderos positivos, que equivale al recall.
- Si la curva se acerca a la esquina superior izquierda, el modelo separa mejor las clases.
- El AUC resume la curva en un solo número: mientras más cerca de 1, mejor.
- Sirve para comparar modelos y entender qué tan bien distinguen entre clases.