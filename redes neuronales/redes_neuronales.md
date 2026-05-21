# Redes neuronales — Guía completa para ingenieros de datos

Esta guía está pensada para llevarte desde los fundamentos teóricos hasta prácticas profesionales para construir, entrenar, evaluar y desplegar redes neuronales. Está escrita en español y orientada a ingenieros de datos que necesitan una referencia completa y práctica.

## Índice
- Introducción y panorama general
- Fundamentos matemáticos
- Componentes de una red neuronal
- Arquitecturas principales
- Preparación de datos y pipelines
- Entrenamiento: pérdidas, optimizadores y estrategias
- Regularización y buenas prácticas
- Arquitecturas modernas y cuándo usarlas
- Métricas y evaluación
- Depuración, experimentación y monitoreo
- Despliegue y MLOps
- Herramientas y ecosistema
- Ejemplos prácticos (PyTorch)
- Checklist práctico y recomendaciones
- Lecturas y recursos

---

## Introducción y panorama general
Las redes neuronales son funciones parametrizadas (por pesos) compuestas de capas lineales y no lineales que se entrenan ajustando dichos parámetros para minimizar una función de pérdida. Se usan en visión por computador, procesamiento de lenguaje natural, series temporales, recomendación, y más.

Objetivos de esta guía:
- Entender la matemática y la intuición.
- Saber qué cambiar en un modelo para mejorar rendimiento.
- Conocer herramientas y patrones para producción.

---

## Fundamentos matemáticos
- Perceptrón: unidad básica, salida = activación(w·x + b).
- Funciones de activación: ReLU, LeakyReLU, Sigmoid, Tanh, GELU. ReLU es estándar en redes profundas por evitar saturación en positivo.
- Propagación hacia adelante (forward): calcular salidas.
- Pérdida (loss): MSE, Cross-Entropy, BCE, Huber, etc.
- Backpropagation: aplicación de la regla de la cadena para obtener gradientes y actualizar parámetros.
- Gradiente descendente y variantes: SGD, SGD+momentum, Adam, AdamW, RMSprop.

Matemática clave:
- Derivadas parciales y jacobiano/gradiente.
- Operaciones matriciales: implementaciones vectorizadas son esenciales para velocidad.

---

## Componentes de una red neuronal
- Capas lineales / convolucionales / recurrentes.
- Normalización: BatchNorm, LayerNorm — estabilizan y aceleran el entrenamiento.
- Dropout: regularización por apagado aleatorio de unidades.
- Activaciones: no linealidad necesaria para aproximar funciones complejas.
- Inicialización de pesos: Xavier/Glorot, He/Kaiming — previenen gradientes que explotan/se desvanecen.

Qué cambiar y dónde:
- Número de neuronas (anchura): aumenta capacidad pero puede requerir más datos/rregularización.
- Número de capas (profundidad): permite representar funciones más complejas.
- Tamaño de lote (`batch_size`): trade-off entre estabilidad y memoria.
- Learning rate (`lr`): el hiperparámetro más crítico; empezar con 1e-3 para Adam, 1e-2/1e-1 para SGD con momentum según escala.
- Optimizer: Adam para prototipos rápidos; SGD+momentum suele generalizar mejor en visión.

---

## Arquitecturas principales
- MLP (Feedforward fully connected): para datos tabulares y como bloque básico.
- CNN (Convolutional Neural Networks): imágenes, audio espectrogramas.
- RNN / LSTM / GRU: secuencias y series temporales (cada vez menos usadas para NLP moderno).
- Transformers: NLP (BERT, GPT), también en visión (ViT).
- Autoencoders, VAEs y GANs: generación y representación latente.

Cuándo elegir qué:
- Imágenes: CNN o ViT (si tienes muchos datos/compute).
- Texto: Transformers.
- Tabular: MLP o modelos boosting (XGBoost) — redes profundas funcionan si hay mucho dato y buen featurizado.

---

## Preparación de datos y pipelines
- Normalizar/estandarizar entradas (mean/std) o usar min-max para imágenes.
- Data augmentation (imágenes): flips, crops, color jitter — mejora generalización.
- Para texto: tokenización, subword (BPE), limpieza mínima.
- Crear Datasets y DataLoaders (PyTorch) o tf.data (TensorFlow).
- Balanceo de clases: oversampling, weighted loss.
- Validación: train/val/test; K-fold para conjuntos pequeños.

---

## Entrenamiento: pérdidas, optimizadores y estrategias
Pérdidas comunes:
- Clasificación multiclase: `CrossEntropyLoss`.
- Regresión: `MSELoss` o `MAELoss`.
- Clasificación binaria: `BCEWithLogitsLoss`.

Optimizadores:
- `Adam`/`AdamW`: aprende rápido, menos tuning inicial.
- `SGD` con momentum + weight decay: suele generalizar bien en visión.

Programas y schedulers:
- `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`.
- Warmup de LR (útil en Transformers).

Técnicas prácticas:
- Early stopping en la métrica de validación.
- Gradiente acumulado si batch_size limitado por GPU.
- Mixed precision (`amp`) para acelerar y reducir memoria en GPUs modernas.

---

## Regularización y buenas prácticas
- Weight decay (L2) aplicado en optimizador o en AdamW.
- Dropout en capas fully-connected para evitar overfitting.
- Data augmentation.
- BatchNorm/LayerNorm para redes profundas.
- Inicialización adecuada de pesos.
- Ensembles y checkpoint averaging (EMA) para producción.

---

## Arquitecturas modernas y cuándo usarlas
- CNNs profundas (ResNet, EfficientNet) para visión; usar transfer learning si hay poco dato.
- Transformers para texto y cada vez más para visión y audio.
- Modelos híbridos: CNN + Transformer.

---

## Métricas y evaluación
- Clasificación: accuracy, precision, recall, F1, AUC-ROC, confusion matrix.
- Regresión: RMSE, MAE, R².
- Evaluación por clases desbalanceadas: usar métricas por clase y macro/micro-averages.
- Calibración de probabilidades: Platt scaling, isotonic regression.

---

## Depuración, experimentación y monitoreo
- Revisar pérdidas y métricas por batch y por época.
- Visualizar predicciones y errores (misclasificados).
- TensorBoard / Weights & Biases para tracking de experimentos.
- Test de gradientes desaparecidos/exploding: monitorizar grad norms.

Errores comunes y cómo detectarlos:
- Pérdida sin bajar: LR muy baja o arquitectura inadecuada; probar aumentar lr y usar scheduler.
- Overfitting: val_loss sube; usar más regularización, augmentation o early stopping.
- Underfitting: aumenta capacidad (más capas/neuronas), reducir dropout.

---

## Despliegue y MLOps
- Serializar modelos: `torch.save` / `state_dict` o `torch.jit.script` / `trace` para optimización.
- Convertir a ONNX para interoperabilidad.
- Servir con TorchServe, TensorFlow Serving, FastAPI + Uvicorn o servicios cloud (SageMaker, Vertex AI).
- Monitorización: latencia, throughput, drift de datos, métricas en producción.
- Pipeline CI/CD: tests de integración, validación de modelos, retrain automatizado.

---

## Herramientas y ecosistema
- PyTorch: flexible, preferido para investigación y desarrollo rápido.
- TensorFlow / Keras: ecosistema amplio y TensorFlow Serving.
- JAX: investigación de alto rendimiento y diferenciación automática avanzada.
- Scikit-learn: preprocesado, pipelines y baselines.
- Librerías auxiliares: torchvision, torchaudio, transformers (Hugging Face), albumentations (augmentation), timm (modelos visión), lightning/accelerate (entrenamiento simplificado).

Instalación rápida (ejemplo pip):
```
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn matplotlib pandas wandb albumentations timm transformers
```

---

## Ejemplos prácticos (PyTorch)
Este ejemplo resume lo que ya tienes en tu notebook `DS-MNIST Simple neural network (numpy pandas pytorch matplotlib).ipynb` y la versión mejorada. Los puntos clave a cambiar para mejorar son: aumentar capas/neurons, usar DataLoader con batching, BatchNorm, Dropout, uso de device (GPU), optimizer y scheduler.

Ejemplo mínimo — MLP entrenable (esqueleto):

```python
# Modelo
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden=[512,256], dropout=0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last,h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last,10))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

# Entrenamiento típico
model = MLP().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
crit = nn.CrossEntropyLoss()
for epoch in range(epochs):
    model.train()
    for xb,yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out,yb)
        loss.backward()
        opt.step()
    # evaluar en validación...
```

Versión comentada línea por línea:

```python
# Importa el módulo principal de PyTorch para construir redes neuronales.
import torch.nn as nn

# Importa el paquete de optimizadores.
import torch.optim as optim

# Define una red neuronal completamente conectada.
class MLP(nn.Module):
    # Constructor: aquí se crean y organizan las capas del modelo.
    def __init__(self, input_dim=784, hidden=[512, 256], dropout=0.2):
        # Inicializa la clase base nn.Module.
        super().__init__()

        # Aquí se guardarán las capas en orden.
        layers = []

        # 'last' guarda el tamaño de la entrada de la siguiente capa.
        last = input_dim

        # Recorre cada tamaño de capa oculta definido en 'hidden'.
        for h in hidden:
            # Crea una capa lineal, una activación ReLU, normalización y dropout.
            layers += [
                nn.Linear(last, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]

            # La salida de esta capa será la entrada de la siguiente.
            last = h

        # Capa final: produce 10 valores, uno por clase del MNIST.
        layers.append(nn.Linear(last, 10))

        # Convierte la lista en una secuencia ejecutable de capas.
        self.net = nn.Sequential(*layers)

    # Define el recorrido de los datos por el modelo.
    def forward(self, x):
        # Devuelve los logits de salida.
        return self.net(x)

# Crea el modelo y lo envía al dispositivo correcto (CPU o GPU).
model = MLP().to(device)

# Pérdida para clasificación multiclase.
crit = nn.CrossEntropyLoss()

# Optimizador Adam con regularización L2 ligera.
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Bucle de entrenamiento por épocas.
for epoch in range(epochs):
    # Activa el modo entrenamiento para BatchNorm y Dropout.
    model.train()

    # Recorre el conjunto de entrenamiento en mini-lotes.
    for xb, yb in train_loader:
        # Mueve datos y etiquetas al mismo dispositivo que el modelo.
        xb, yb = xb.to(device), yb.to(device)

        # Limpia gradientes de iteraciones anteriores.
        opt.zero_grad()

        # Forward pass: el modelo produce predicciones.
        out = model(xb)

        # Calcula el error entre la salida y la etiqueta real.
        loss = crit(out, yb)

        # Backpropagation: calcula gradientes respecto a cada peso.
        loss.backward()

        # Actualiza los parámetros usando Adam.
        opt.step()

    # En un flujo real, aquí evaluarías en validación.
    # evaluar en validación...
```

Explicación de cada línea importante:
- `import torch.nn as nn`: te da acceso a capas, pérdidas y módulos.
- `import torch.optim as optim`: te da acceso a optimizadores como Adam y SGD.
- `class MLP(nn.Module)`: crea una clase de modelo reutilizable.
- `input_dim=784`: cada imagen MNIST se aplana a 28 x 28 = 784 valores.
- `hidden=[512, 256]`: define el tamaño de las capas ocultas.
- `dropout=0.2`: apaga el 20% de neuronas durante entrenamiento.
- `super().__init__()`: inicializa internamente la clase base de PyTorch.
- `layers = []`: prepara una lista donde se guardarán las capas.
- `last = input_dim`: indica cuántas entradas recibe la primera capa.
- `nn.Linear(last, h)`: aprende una transformación lineal entre capas.
- `nn.ReLU()`: introduce no linealidad; sin ella, la red sería mucho más limitada.
- `nn.BatchNorm1d(h)`: estabiliza el entrenamiento y ayuda a converger mejor.
- `nn.Dropout(dropout)`: reduce overfitting obligando a no depender de neuronas específicas.
- `layers.append(nn.Linear(last, 10))`: produce una salida por clase.
- `self.net = nn.Sequential(*layers)`: encadena todo en orden.
- `forward(self, x)`: define cómo avanza el dato dentro del modelo.
- `return self.net(x)`: devuelve los logits sin softmax, porque `CrossEntropyLoss` ya lo maneja.
- `model = MLP().to(device)`: mueve el modelo a GPU si existe.
- `CrossEntropyLoss()`: compara logits con etiquetas enteras de clase.
- `Adam(..., lr=1e-3, weight_decay=1e-5)`: optimizador con tasa de aprendizaje y regularización.
- `model.train()`: activa comportamiento de entrenamiento.
- `train_loader`: divide los datos en mini-lotes para entrenar de forma eficiente.
- `xb, yb = xb.to(device), yb.to(device)`: evita errores de CPU/GPU mezclados.
- `opt.zero_grad()`: evita acumular gradientes de pasos previos.
- `out = model(xb)`: hace el pase hacia adelante.
- `loss = crit(out, yb)`: calcula cuánto se equivoca la red.
- `loss.backward()`: calcula la señal de ajuste para cada peso.
- `opt.step()`: actualiza los pesos.

Qué cambiar para mejorar resultados:
- Más neuronas: aumenta `hidden` a algo como `[1024, 512, 256]`.
- Más capas: añade más valores en `hidden`.
- Más o menos dropout: si ves sobreajuste, súbelo; si ves subajuste, bájalo.
- Más `batch_size`: puede estabilizar el gradiente, pero consume más memoria.
- Mejor `lr`: si la pérdida baja muy poco, prueba `3e-3` o `1e-3` según el caso.

Si quieres, el siguiente paso natural es convertir este ejemplo en una CNN y explicarte también cada línea de esa versión, porque para MNIST suele rendir mejor que una MLP.

Consejos prácticos:
- Si la pérdida baja poco: aumentar lr (por ejemplo 1e-3 → 3e-3), añadir capas o neuronas, usar batchnorm, o cambiar a un optimizador diferente.
- Observa la curva de `train_loss` vs `val_loss` para saber si hay under/overfitting.

---

## Checklist práctico para mejorar tu modelo (rápido)
- [ ] ¿Usas batching? Si no, añade DataLoader con `batch_size`=64-256.
- [ ] ¿Normalizas los datos? Sí: `X = (X - mean) / std`.
- [ ] ¿Usas BatchNorm/LayerNorm en redes profundas? Añade si no.
- [ ] ¿Tienes Dropout excesivo? Reduce si underfitting.
- [ ] ¿LR correcto? Haz sweeps y usa schedulers.
- [ ] ¿Usas GPU y mixed precision? Mejora velocidad y capacidad.
- [ ] ¿Haces augmentation (imágenes)? Sí, aumenta generalización.

---

## Recomendaciones profesionales
- Versiona experimentos (W&B, MLflow) y guarda `state_dict` y hyperparams en cada run.
- Automatiza evaluaciones y pruebas de regresión de modelos.
- Prefiere `torch.save(model.state_dict())` para checkpoints y `torch.jit.script` para inferencia optimizada.
- Para visión, usar transfer learning de `torchvision.models` con fine-tuning suele ahorrar tiempo y datos.

---

## Dónde cambiar cosas en tus notebooks
- En `DS-MNIST Simple neural network (numpy pandas pytorch matplotlib).ipynb` cambia la definición de `Net` por un `MLP` más profundo o una `CNN`.
- En `DS-MNIST Simple neural network (improved).ipynb` revisa: `batch_size` en DataLoader, `lr` en `optim.Adam`, `scheduler` y las capas dentro de `Net`.

Archivos relevantes en tu workspace:
- [redes neuronales/DS-MNIST Simple neural network (numpy pandas pytorch matplotlib).ipynb](redes%20neuronales/DS-MNIST%20Simple%20neural%20network%20%28numpy%20pandas%20pytorch%20matplotlib%29.ipynb)
- [redes neuronales/DS-MNIST Simple neural network (improved).ipynb](redes%20neuronales/DS-MNIST%20Simple%20neural%20network%20%28improved%29.ipynb)

---

## Lecturas y recursos
- Deep Learning (Goodfellow, Bengio, Courville) — teoría.
- Ian Goodfellow, Adam Paszke (pytorch tutorials).
- Hugging Face course — transformers.
- Fast.ai course — práctica y transfer learning.
- Papers: ResNet, Transformer, Attention is All You Need.

---

## Siguientes pasos sugeridos
- Probar la `improved` notebook y ajustar `lr`, `batch_size` y arquitectura.
- Si quieres, convierto el `MLP` a una `CNN` (recomendado para MNIST) y lo dejo listo para entrenamiento con buenas prácticas (mixed precision, checkpointing, W&B).

---

Si quieres, ahora:
- Puedo añadir ejemplos de CNN y un script `train.py` listo para producción.
- O puedo integrar seguimiento de experimentos con Weights & Biases.

