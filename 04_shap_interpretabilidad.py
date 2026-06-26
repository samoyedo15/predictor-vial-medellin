# ============================================================
# SCRIPT 4: Interpretabilidad con SHAP Values
# Requiere: gravedad_model.pkl (generado por 02_entrenamiento.py)
# ============================================================
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from preparacion_datos import pipeline_completo

plt.rcParams.update({'figure.figsize': (10, 8), 'font.size': 13, 'figure.dpi': 110})

# ── Cargar datos y modelo ──
print("Ejecutando pipeline de datos...")
X_train, X_test, y_train, y_test, feature_names = pipeline_completo()

print("Cargando modelo...")
modelo = joblib.load('gravedad_model.pkl')
print(f"✓ Modelo cargado: {type(modelo).__name__}")

# ── Calcular SHAP values ──
explainer    = shap.TreeExplainer(modelo)
X_test_muestra = X_test.sample(n=min(500, len(X_test)), random_state=42)

print(f"\nCalculando SHAP values para {len(X_test_muestra)} muestras...")
shap_values = explainer.shap_values(X_test_muestra)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

# ── Gráfico 1: Importancia global (barras) ──
print("\nGráfico 1: Importancia global SHAP...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, X_test_muestra, plot_type="bar", max_display=15, show=False)
plt.title("Importancia de Variables (SHAP) — Top 15", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ── Gráfico 2: Beeswarm (dirección del impacto) ──
print("Gráfico 2: SHAP Beeswarm (dirección del impacto)...")
fig2, ax2 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, X_test_muestra, max_display=15, show=False)
plt.title("Impacto de Variables en la Predicción (SHAP)", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ── Gráfico 3: Waterfall para un incidente individual ──
print("Gráfico 3: Explicación individual (waterfall)...")
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(base_val[1])

vals_incidente = shap_vals[0]
if vals_incidente.ndim == 2:
    vals_incidente = vals_incidente[:, 1]

shap.plots.waterfall(shap.Explanation(
    values=vals_incidente,
    base_values=base_val,
    data=X_test_muestra.iloc[0],
    feature_names=list(X_test_muestra.columns)
), show=False)
plt.title("Explicación SHAP — Incidente Individual", fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

print("\n✓ Análisis SHAP completado.")
print("  Barras     → qué variables importan más EN GENERAL")
print("  Beeswarm   → la DIRECCIÓN del impacto (rojo = valor alto)")
print("  Waterfall  → POR QUÉ el modelo predijo así para UN incidente específico")
