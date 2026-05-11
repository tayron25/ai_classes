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
