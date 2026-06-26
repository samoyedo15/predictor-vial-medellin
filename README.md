# 🚦 Predictor de Gravedad de Incidentes Viales — Medellín

Sistema de inteligencia artificial que predice si un incidente vial en Medellín resultará en **víctimas (heridos o fallecidos)** o en **solo daños materiales**, con explicación SHAP por predicción.

> Desarrollado por **Anderson Marin** — Curso de Profundización en Inteligencia Artificial · 2026  
> Dataset: [Incidentes Viales Medellín — datos.gov.co](https://www.datos.gov.co/Transporte/Accidentes-Viales/yu3i-jau4)

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Demo](#demo)
- [Características](#características)
- [Tecnologías](#tecnologías)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación y Ejecución](#instalación-y-ejecución)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Modelo y Métricas](#modelo-y-métricas)
- [Dataset](#dataset)
- [Limitaciones](#limitaciones)
- [Licencia](#licencia)

---

## Descripción del Proyecto

El sistema fue entrenado con **255,804 incidentes reales** registrados entre 2014 y 2022 por la Secretaría de Movilidad de la Alcaldía de Medellín. A partir de variables contextuales como la hora del incidente, el tipo de accidente, el diseño de la vía y la ubicación por comuna, el modelo predice la probabilidad de que el evento resulte en víctimas.

**Problema que resuelve:** la evaluación del riesgo de un incidente vial era un proceso manual y subjetivo. Este sistema lo convierte en una predicción objetiva, explicable y trazable en segundos.

**A quién impacta:**
- Secretaría de Movilidad y entidades de seguridad vial
- Equipos de análisis urbano y planificación de recursos de emergencia
- Investigadores en movilidad y seguridad vial

---

## Demo

```bash
# Clonar el repositorio
git clone https://github.com/USUARIO/predictor-vial-medellin.git
cd predictor-vial-medellin

# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo (genera gravedad_model.pkl)
python 02_entrenamiento.py

# Lanzar la aplicación
streamlit run app_vial.py
```

La app se abre automáticamente en `http://localhost:8501`

---

## Características

| Funcionalidad | Descripción |
|---|---|
| **Predicción Individual** | Formulario interactivo para evaluar un incidente específico |
| **Predicción por Lote** | Carga de CSV con múltiples incidentes y exportación de resultados |
| **Explicación SHAP** | Gráfico de barras que explica qué variables influyeron en cada predicción |
| **Segmentación** | Análisis de 5,000 incidentes históricos por nivel de riesgo |
| **Comparación de Modelos** | Tabla y gráfico comparativo: Logistic Regression vs Random Forest vs XGBoost |
| **Dashboard** | Resumen ejecutivo del modelo con métricas, hallazgos clave y flujo del sistema |

---

## Tecnologías

```
Python 3.10+
├── scikit-learn     — Random Forest, métricas de evaluación
├── pandas / numpy   — Manipulación de datos
├── shap             — Interpretabilidad SHAP (TreeExplainer)
├── streamlit        — Interfaz web
├── plotly           — Visualizaciones interactivas
├── joblib           — Serialización del modelo (.pkl)
├── matplotlib / seaborn — Gráficos de exploración y evaluación
└── xgboost          — Modelo alternativo en comparación
```

---

## Estructura del Proyecto

```
predictor-vial-medellin/
│
├── app_vial.py                     # Router principal de Streamlit
├── constants.py                    # Listas de categorías (CLASES, DISENOS, COMUNAS)
├── styles.py                       # CSS de la interfaz
├── utils.py                        # Funciones reutilizables del modelo
├── preparacion_datos.py            # Pipeline de carga y preprocesamiento
│
├── paginas/                        # Módulos de cada página de la app
│   ├── __init__.py
│   ├── dashboard.py
│   ├── prediccion_individual.py
│   ├── prediccion_lote.py
│   ├── segmentacion.py
│   ├── comparacion_modelos.py
│   └── acerca_del_modelo.py
│
├── 01_exploracion.py               # EDA: exploración y visualizaciones del dataset
├── 02_entrenamiento.py             # Entrena Random Forest, genera los .pkl
├── 03_comparacion_modelos.py       # Compara los 3 algoritmos + cross-validation
├── 04_shap_interpretabilidad.py    # Análisis SHAP global (summary plot, beeswarm, waterfall)
│
├── gravedad_model.pkl              # Modelo entrenado (generado por 02_entrenamiento.py)
├── feature_names_vial.pkl          # Columnas del modelo (generado por 02_entrenamiento.py)
├── Incidentes-Viales-Medellin.csv  # Dataset (descargado automáticamente)
│
├── requirements.txt
└── README.md
```

---

## Instalación y Ejecución

### Prerequisitos

- Python 3.10 o superior
- pip

### Paso 1 — Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 2 — Entrenar el modelo

```bash
python 02_entrenamiento.py
```

Este script:
1. Carga el dataset `Incidentes-Viales-Medellin.csv`
2. Aplica feature engineering y limpieza de datos
3. Entrena el Random Forest (200 árboles, ~3–5 minutos)
4. Evalúa el modelo y muestra métricas
5. Guarda `gravedad_model.pkl` y `feature_names_vial.pkl`

### Paso 3 — Lanzar la aplicación

```bash
streamlit run app_vial.py
```

### Scripts adicionales (opcionales)

```bash
# Exploración y visualización del dataset
python 01_exploracion.py

# Comparar Logistic Regression, Random Forest y XGBoost
python 03_comparacion_modelos.py

# Análisis SHAP global (summary plot, beeswarm, waterfall)
python 04_shap_interpretabilidad.py
```

---

## Uso de la Aplicación

### Predicción Individual

1. Ir a **🔍 Predicción Individual** en el menú lateral
2. Seleccionar **clase de accidente**, **diseño de vía** y **comuna**
3. Ajustar **hora**, **día** y **mes** con los controles deslizantes
4. El sistema calcula automáticamente el **periodo del día** y si es **fin de semana**
5. Hacer clic en **Predecir Gravedad**

**Resultados que obtendrás:**
- **Medidor de probabilidad** (0–100%) con código de colores: 🟢 Bajo / 🟡 Medio / 🔴 Alto
- **Plan de prevención** con recomendaciones contextuales según el tipo de incidente
- **Gráfico SHAP** con las 12 variables que más influyeron (rojo = aumentó riesgo, azul = lo redujo)

### Predicción por Lote

1. Ir a **📁 Predicción por Lote**
2. Descargar la **plantilla CSV de ejemplo**
3. Completar el CSV con tus incidentes
4. Subir el archivo y hacer clic en **Ejecutar predicciones**
5. Descargar los resultados con probabilidad y nivel de riesgo por incidente

**Formato del CSV:**

```csv
CLASE,DISENO,COMUNA,MES,hora,dia_semana
Atropello,Interseccion,La Candelaria,9,22,4
Choque,Tramo de via,Castilla,3,8,1
```

---

## Modelo y Métricas

### Comparación de Algoritmos

| Modelo | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7706 | 0.7446 | 0.8406 |
| **Random Forest** ✅ | **0.7696** | **0.7399** | **0.8440** |
| XGBoost | 0.7728 | 0.7505 | 0.8469 |

*Evaluación sobre 51,161 incidentes del conjunto de prueba (20% estratificado)*

### Hiperparámetros del Modelo Seleccionado

```python
RandomForestClassifier(
    n_estimators=200,      # 200 árboles en el ensamble
    max_depth=12,          # Profundidad máxima por árbol
    class_weight='balanced',  # Compensa el desbalance 59/41%
    random_state=42,       # Reproducibilidad
    n_jobs=-1              # Todos los núcleos del CPU
)
```

### ¿Por qué Random Forest sobre XGBoost?

Aunque XGBoost obtuvo el mayor ROC-AUC (0.847 vs 0.844), la diferencia es marginal. Random Forest fue seleccionado por:

- **SHAP con TreeExplainer**: interpretación directa en probabilidad (no en log-odds como XGBoost)
- **`class_weight='balanced'`**: compensa el desbalance 59/41% sin ajustes finos adicionales
- **Sin hiperparámetros críticos**: no hay sensibilidad a `learning_rate` ni `gamma`
- **`n_jobs=-1`**: paraleliza eficientemente sobre 256K registros

### Variables del Modelo

| Variable | Tipo | Descripción |
|---|---|---|
| `hora` | Numérica (0–23) | Hora del incidente |
| `dia_semana` | Numérica (0–6) | 0 = Lunes … 6 = Domingo |
| `es_fin_de_semana` | Binaria | Calculada automáticamente |
| `MES` | Numérica (1–12) | Mes del año |
| `periodo` | Categórica | Madrugada / Mañana / Tarde / Noche |
| `CLASE_ACCIDENTE` | Categórica (6 val) | Tipo de accidente |
| `DISEÑO` | Categórica (11 val) | Tipo de vía |
| `COMUNA` | Categórica (21 val) | Ubicación geográfica |

Variables descartadas: `BARRIO` (300+ categorías), `DIRECCION` (texto libre), `EXPEDIENTE`, coordenadas X/Y.

### Interpretabilidad SHAP

Cada predicción individual incluye un gráfico SHAP que responde: *"¿Por qué el modelo predijo este resultado para este incidente específico?"*

- **Barras rojas** → esa variable aumentó la probabilidad de víctimas
- **Barras azules** → esa variable redujo la probabilidad de víctimas
- **Longitud de barra** → magnitud de la influencia

---

## Dataset

| Campo | Valor |
|---|---|
| Fuente | Secretaría de Movilidad — Alcaldía de Medellín |
| Portal | [datos.gov.co](https://www.datos.gov.co/Transporte/Accidentes-Viales/yu3i-jau4) |
| Período | 2014 – 2022 (8 años) |
| Registros | 255,804 incidentes |
| Variable objetivo | `victimas`: Con víctimas (59.2%) / Solo daños (40.8%) |
| Licencia | Datos Abiertos — Ley 1712 de 2014 |
| Datos personales | Ninguno — sin nombres, cédulas ni matrículas |

---

## Limitaciones

- **Solo Medellín**: el modelo no generaliza a otras ciudades sin reentrenamiento
- **Datos hasta 2022**: cambios de infraestructura post-2022 no están capturados
- **Fallecidos agrupados**: no distingue heridos leves de fatales (~0.13% son fatales)
- **Sin variables climáticas**: lluvia y neblina no están incluidas
- **Resolución de comuna**: no identifica puntos negros específicos dentro de una zona
- **Herramienta de apoyo**: el modelo provee estimaciones probabilísticas, no certezas; no debe reemplazar el criterio experto

---

## Ética

- Dataset 100% público bajo Ley 1712 de 2014 (Transparencia y Acceso a la Información Pública)
- Sin información personal identificable
- Fuente oficial: Secretaría de Movilidad de la Alcaldía de Medellín
- Uso académico — Curso de Profundización en IA

---

## Licencia

Este proyecto es de uso académico. El dataset utilizado está bajo licencia de datos abiertos del Estado colombiano (Ley 1712 de 2014).

---

*Desarrollado por Anderson Marin · Curso de Profundización en Inteligencia Artificial · 2026*
