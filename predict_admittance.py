import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
df = pd.read_csv('Admittance.csv')
df['Admitted_num'] = df['Admitted'].map({'No': 0, 'Yes': 1})

# Preparar datos
X = df[['SAT']]
y = df['Admitted_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar Regresión Logística
logreg_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])
logreg_model.fit(X_train, y_train)

# Entrenar Random Forest
rf_model = Pipeline(steps=[
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])
rf_model.fit(X_train, y_train)

print("=" * 60)
print("PREDICTOR DE ADMISIÓN UNIVERSITARIA")
print("=" * 60)

while True:
    try:
        sat_input = input("\nIngresa tu puntaje SAT (o 'salir' para terminar): ")
        
        if sat_input.lower() == 'salir':
            print("\n¡Hasta luego!")
            break
        
        sat_value = float(sat_input)
        sat_df = pd.DataFrame({'SAT': [sat_value]})
        
        # Predicciones
        logreg_prob = logreg_model.predict_proba(sat_df)[0][1]
        rf_prob = rf_model.predict_proba(sat_df)[0][1]
        
        logreg_pred = "Probablemente entrarás" if logreg_prob >= 0.5 else "Probablemente no entrarás"
        rf_pred = "Probablemente entrarás" if rf_prob >= 0.5 else "Probablemente no entrarás"
        
        print(f"\n{'─' * 60}")
        print(f"Puntaje SAT: {sat_value}")
        print(f"{'─' * 60}")
        print(f"\n🔹 REGRESIÓN LOGÍSTICA:")
        print(f"   {logreg_pred}")
        print(f"   Probabilidad: {logreg_prob:.1%}")
        print(f"\n🔹 RANDOM FOREST:")
        print(f"   {rf_pred}")
        print(f"   Probabilidad: {rf_prob:.1%}")
        print(f"{'─' * 60}")
        
    except ValueError:
        print("❌ Por favor ingresa un número válido para el SAT.")
