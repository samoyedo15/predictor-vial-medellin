# ============================================================
# SCRIPT 2: Entrenamiento y Evaluación del Modelo Random Forest
# Genera: gravedad_model.pkl, feature_names_vial.pkl
# ============================================================
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix, roc_curve)

from preparacion_datos import pipeline_completo

sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (12, 7), 'font.size': 13,
                     'axes.titlesize': 15, 'figure.dpi': 110})

# ── Preparar datos ──
print("Ejecutando pipeline de datos...")
X_train, X_test, y_train, y_test, feature_names = pipeline_completo()

print(f"Entrenamiento: {X_train.shape[0]:,} incidentes  |  Prueba: {X_test.shape[0]:,} incidentes")
print(f"Features: {len(feature_names)}")
print(f"Proporción víctimas (train): {y_train.mean()*100:.1f}%")
print(f"Proporción víctimas (test):  {y_test.mean()*100:.1f}%")

# ── Entrenar ──
print("\nEntrenando Random Forest...")
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
modelo.fit(X_train, y_train)
print("✓ Modelo entrenado")

# ── Evaluar ──
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("MÉTRICAS DE EVALUACIÓN")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}  ({accuracy*100:.1f}% de predicciones correctas)")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=['Solo daños', 'Con víctimas']))

# ── Matriz de Confusión ──
cm_val  = confusion_matrix(y_test, y_pred)
total   = cm_val.sum()
annot   = np.array([[f"{cm_val[i,j]:,}\n({cm_val[i,j]/total*100:.1f}%)"
                     for j in range(2)] for i in range(2)])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_val, annot=annot, fmt='', cmap='Blues',
            xticklabels=['Solo daños', 'Con víctimas'],
            yticklabels=['Solo daños', 'Con víctimas'],
            ax=ax, linewidths=0.5, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'})
ax.set_title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=14)
ax.set_ylabel('Valor Real'); ax.set_xlabel('Predicción del Modelo')
plt.tight_layout(); plt.show()

# ── Curva ROC ──
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.fill_between(fpr, tpr, alpha=0.12, color='#2980b9')
ax.plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'Random Forest — AUC = {roc_auc:.4f}')
ax.plot([0, 1], [0, 1], color='#888', linestyle='--', lw=1.5, label='Clasificador aleatorio (AUC = 0.50)')
ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
ax.set_title('Curva ROC — Capacidad Discriminante del Modelo', fontsize=16, fontweight='bold', pad=14)
ax.legend(loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
plt.tight_layout(); plt.show()

# ── Importancia de Variables ──
importancias = pd.Series(modelo.feature_importances_, index=feature_names)

top_15       = importancias.sort_values(ascending=True).tail(15)

norm_vals   = (top_15.values - top_15.values.min()) / (top_15.values.max() - top_15.values.min() + 1e-9)
colores_imp = [cm.Blues(0.4 + 0.5 * v) for v in norm_vals]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_15)), top_15.values, color=colores_imp, edgecolor='white', linewidth=0.3)
ax.set_yticks(range(len(top_15))); ax.set_yticklabels(top_15.index, fontsize=11)
ax.set_title('Top 15 Variables Más Importantes', fontsize=16, fontweight='bold', pad=14)
ax.set_xlabel('Importancia (Gini)'); ax.spines[['top', 'right']].set_visible(False)
for bar, val in zip(bars, top_15.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout(); plt.show()

# ── Guardar modelo ──
joblib.dump(modelo, 'gravedad_model.pkl')
joblib.dump(feature_names, 'feature_names_vial.pkl')
print("\n✓ gravedad_model.pkl guardado")
print("✓ feature_names_vial.pkl guardado")

# ── Prueba rápida ──
modelo_cargado = joblib.load('gravedad_model.pkl')
print("✓ Verificación: modelo se carga correctamente")
print("\nEjecuta 03_comparacion_modelos.py para comparar con otros algoritmos.")
print("Ejecuta 04_shap_interpretabilidad.py para análisis SHAP.")
