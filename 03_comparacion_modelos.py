# ============================================================
# SCRIPT 3: Comparación de Modelos (Logistic Regression, Random Forest, XGBoost)
# Requiere haber ejecutado 02_entrenamiento.py al menos una vez
# para confirmar que el pipeline funciona.
# ============================================================
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from preparacion_datos import pipeline_completo

plt.rcParams.update({'figure.figsize': (15, 5), 'font.size': 13, 'figure.dpi': 110})

# ── Preparar datos ──
print("Ejecutando pipeline de datos...")
X_train, X_test, y_train, y_test, feature_names = pipeline_completo()
print(f"Datos listos. Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── Definir modelos ──
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw   = n_neg / n_pos

modelos = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, random_state=42,
        eval_metric='logloss', n_jobs=-1)
}

# ── Entrenar y evaluar ──
print("\n" + "=" * 60)
print("COMPARACIÓN DE MODELOS")
print("=" * 60)

resultados = []
for nombre, mod in modelos.items():
    print(f"\nEntrenando {nombre}...")
    mod.fit(X_train, y_train)
    y_p  = mod.predict(X_test)
    y_pr = mod.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_p)
    f1_val  = f1_score(y_test, y_p)
    roc_val = roc_auc_score(y_test, y_pr)

    resultados.append({
        'Modelo':   nombre,
        'Accuracy': round(acc, 4),
        'F1-Score': round(f1_val, 4),
        'ROC-AUC':  round(roc_val, 4)
    })
    print(f"  Accuracy: {acc:.4f}  |  F1: {f1_val:.4f}  |  ROC-AUC: {roc_val:.4f}")

df_comp = pd.DataFrame(resultados)
print("\n" + "=" * 60)
print("TABLA COMPARATIVA")
print("=" * 60)
print(df_comp.to_string(index=False))

# ── Gráfico comparativo ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colores   = ['#3498db', '#2ecc71', '#e74c3c']
metricas  = ['Accuracy', 'F1-Score', 'ROC-AUC']

for i, metrica in enumerate(metricas):
    axes[i].bar(df_comp['Modelo'], df_comp[metrica], color=colores)
    axes[i].set_title(metrica, fontsize=14)
    axes[i].set_ylim(0.4, 1.0)
    axes[i].tick_params(axis='x', rotation=25)
    for j, v in enumerate(df_comp[metrica]):
        axes[i].text(j, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

plt.suptitle('Comparación de Modelos — Conjunto de Prueba (20%)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ── Cross-Validation ──
print("\n" + "=" * 60)
print("CROSS-VALIDATION (5 FOLDS) — puede tardar varios minutos")
print("=" * 60)

for nombre, mod in modelos.items():
    cv_acc = cross_val_score(mod, X_train, y_train, cv=5, scoring='accuracy',  n_jobs=-1)
    cv_f1  = cross_val_score(mod, X_train, y_train, cv=5, scoring='f1',        n_jobs=-1)
    cv_roc = cross_val_score(mod, X_train, y_train, cv=5, scoring='roc_auc',   n_jobs=-1)
    print(f"\n{nombre}:")
    print(f"  Accuracy:  {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  ROC-AUC:   {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

# ── Guardar Random Forest como modelo final ──
mejor_modelo = modelos['Random Forest']
mejor_modelo.fit(X_train, y_train)
joblib.dump(mejor_modelo, 'gravedad_model.pkl')
joblib.dump(feature_names, 'feature_names_vial.pkl')
print("\n✓ Modelo final (Random Forest) guardado en gravedad_model.pkl")
print("  Justificación: mejor soporte SHAP + class_weight='balanced' + sin hiperparámetros críticos")
