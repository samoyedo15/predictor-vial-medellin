# ============================================================
# SCRIPT 1: Exploración y Análisis Exploratorio de Datos (EDA)
# Ejecutar antes de entrenar el modelo.
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

from preparacion_datos import cargar_datos, crear_features_temporales

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (12, 7), 'font.size': 13,
    'axes.titlesize': 15, 'axes.labelsize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'figure.dpi': 110
})

# ── Cargar datos ──
print("Cargando dataset...")
df = cargar_datos()
print(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"\nColumnas: {list(df.columns)}\n")

print("=== TIPOS DE DATOS ===")
print(df.dtypes)

print("\n=== VALORES NULOS ===")
nulos = df.isnull().sum()
print(nulos[nulos > 0].sort_values(ascending=False))

# ── Variable objetivo ──
print("\n=== DISTRIBUCIÓN DE GRAVEDAD ===")
print(df['GRAVEDAD_ACCIDENTE'].value_counts())
print()
print(df['GRAVEDAD_ACCIDENTE'].value_counts(normalize=True).round(4) * 100)

fig, ax = plt.subplots(figsize=(9, 6))
conteos = df['GRAVEDAD_ACCIDENTE'].value_counts()
conteos.plot(kind='bar', color=['#f39c12', '#2ecc71', '#e74c3c'], ax=ax,
             edgecolor='white', linewidth=0.5)
ax.set_title('Distribución de Gravedad de Incidentes Viales', fontsize=16, fontweight='bold', pad=14)
ax.set_xlabel('Gravedad'); ax.set_ylabel('Cantidad de incidentes')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.spines[['top', 'right']].set_visible(False)
total = conteos.sum()
for i, v in enumerate(conteos.values):
    ax.text(i, v + total * 0.01, f"{v:,}\n({v/total*100:.1f}%)", ha='center', fontweight='bold')
plt.tight_layout(); plt.show()

# ── Variables categóricas ──
print("\n=== CLASE DE ACCIDENTE ===")
print(df['CLASE_ACCIDENTE'].value_counts())
print("\n=== DISEÑO DE VÍA ===")
print(df['DISEÑO'].value_counts())
print("\n=== TOP 10 COMUNAS ===")
print(df['COMUNA'].value_counts().head(10))
print("\n=== ESTADÍSTICAS NUMÉRICAS ===")
print(df.describe())

# ── Visualización exploratoria ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

top_clases   = df['CLASE_ACCIDENTE'].value_counts().head(5).index
df_top_cl    = df[df['CLASE_ACCIDENTE'].isin(top_clases)]
sns.countplot(data=df_top_cl, x='CLASE_ACCIDENTE', hue='GRAVEDAD_ACCIDENTE',
              palette=['#f39c12', '#2ecc71', '#e74c3c'], ax=axes[0, 0])
axes[0, 0].set_title('Gravedad por Clase de Accidente')
axes[0, 0].tick_params(axis='x', rotation=30)

sns.countplot(data=df, x='MES', palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Incidentes por Mes')

top_disenos  = df['DISEÑO'].value_counts().head(5).index
df_top_dis   = df[df['DISEÑO'].isin(top_disenos)]
sns.countplot(data=df_top_dis, x='DISEÑO', hue='GRAVEDAD_ACCIDENTE',
              palette=['#f39c12', '#2ecc71', '#e74c3c'], ax=axes[1, 0])
axes[1, 0].set_title('Gravedad por Diseño de Vía')
axes[1, 0].tick_params(axis='x', rotation=30)

top_comunas = df['COMUNA'].value_counts().head(10)
top_comunas.sort_values().plot(kind='barh', color='#2980b9', ax=axes[1, 1])
axes[1, 1].set_title('Top 10 Comunas con Más Incidentes')

plt.tight_layout(); plt.show()

# ── Features temporales ──
df = crear_features_temporales(df)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df, x='hora', palette='magma', ax=axes[0, 0])
axes[0, 0].set_title('Incidentes por Hora del Día')

sns.countplot(data=df, x='periodo', hue='GRAVEDAD_ACCIDENTE',
              palette=['#f39c12', '#2ecc71', '#e74c3c'],
              order=['Madrugada', 'Mañana', 'Tarde', 'Noche'], ax=axes[0, 1])
axes[0, 1].set_title('Gravedad por Periodo del Día')

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
conteo_dias = df['dia_semana'].value_counts().sort_index()
axes[1, 0].bar(dias, conteo_dias.values, color='#3498db')
axes[1, 0].set_title('Incidentes por Día de la Semana')

sns.countplot(data=df, x='es_fin_de_semana', hue='GRAVEDAD_ACCIDENTE',
              palette=['#f39c12', '#2ecc71', '#e74c3c'], ax=axes[1, 1])
axes[1, 1].set_title('Gravedad: Entre semana vs Fin de semana')
axes[1, 1].set_xticklabels(['Entre semana', 'Fin de semana'])

plt.tight_layout(); plt.show()

print("\n✓ Exploración completada.")
print("  Ejecuta 02_entrenamiento.py para entrenar el modelo.")
