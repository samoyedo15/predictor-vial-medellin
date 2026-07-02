# 🚦 Predicción de Gravedad de Incidentes Viales — Medellín

Sistema de inteligencia artificial que predice si un incidente vial en Medellín resultará en **víctimas (heridos/fallecidos)** o **solo daños materiales**, a partir del contexto del incidente (hora, tipo de accidente, diseño de la vía y comuna). Incluye una aplicación interactiva en **Streamlit** con explicabilidad **SHAP** y recomendaciones de prevención.

> Proyecto desarrollado por **Anderson Marin** — Curso de Profundización en Inteligencia Artificial.

### 🌐 [Ver la aplicación en vivo →](https://predictor-vial-medellin.streamlit.app/)

---

## 📌 Tabla de contenido

- [Descripción del problema](#-descripción-del-problema)
- [Demo de la aplicación](#-demo-de-la-aplicación)
- [Dataset](#-dataset)
- [Arquitectura del proyecto](#-arquitectura-del-proyecto)
- [Modelo de machine learning](#-modelo-de-machine-learning)
- [Resultados](#-resultados)
- [Interpretabilidad con SHAP](#-interpretabilidad-con-shap)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Instalación y uso](#-instalación-y-uso)
- [Funcionalidades de la app](#-funcionalidades-de-la-app)
- [Formato de CSV para predicción por lote](#-formato-de-csv-para-predicción-por-lote)
- [Limitaciones](#-limitaciones)
- [Ética y privacidad de los datos](#-ética-y-privacidad-de-los-datos)
- [Tecnologías utilizadas](#-tecnologías-utilizadas)
- [Documentación adicional](#-documentación-adicional)
- [Autor](#-autor)

---

## 🎯 Descripción del problema

La Secretaría de Movilidad de Medellín registra miles de incidentes viales cada año, pero no existe una forma rápida y cuantitativa de estimar, en el momento del reporte, **qué tan probable es que un incidente tenga víctimas**. Esto dificulta la priorización de recursos preventivos (señalización, iluminación, control de velocidad, presencia de agentes de tránsito) hacia los contextos de mayor riesgo.

Este proyecto entrena un modelo de **clasificación binaria supervisada** sobre 255.804 incidentes reales (2014–2022) para predecir la probabilidad de que un incidente vial tenga víctimas, y expone esa predicción a través de una aplicación web con explicaciones interpretables y recomendaciones de prevención.

---

## 🖥️ Demo de la aplicación

🔗 **URL pública:** [predictor-vial-medellin.streamlit.app](https://predictor-vial-medellin.streamlit.app/)

La aplicación cuenta con 6 secciones:

| Sección | Descripción |
|---|---|
| 🏠 **Dashboard** | Resumen ejecutivo, métricas del modelo y hallazgos clave |
| 🔍 **Predicción Individual** | Evalúa un incidente puntual con explicación SHAP en tiempo real |
| 📁 **Predicción por Lote** | Carga un CSV con múltiples incidentes y exporta resultados |
| 📊 **Segmentación** | Explora 5.000 incidentes históricos segmentados por riesgo |
| 🔬 **Comparación de Modelos** | Random Forest vs. XGBoost vs. Regresión Logística |
| ℹ️ **Acerca del Modelo** | Metodología, datos, arquitectura y limitaciones |

---

## 📊 Dataset

| Atributo | Detalle |
|---|---|
| Fuente | Secretaría de Movilidad — Alcaldía de Medellín ([datos.gov.co](https://www.datos.gov.co/Transporte/Accidentes-Viales/yu3i-jau4)) |
| Registros | 255.804 incidentes |
| Periodo | 2014 – 2022 |
| Licencia | Datos abiertos — Ley 1712 de 2014 |
| Variable objetivo | `victimas`: `1` = con heridos/fallecidos (59,2%) · `0` = solo daños (40,8%) |
| Información personal | Ninguna (sin nombres, cédulas ni placas) |

**Variables utilizadas por el modelo:**

- ⏰ **Temporales**: `hora`, `dia_semana`, `es_fin_de_semana`, `MES`, `periodo` (Madrugada/Mañana/Tarde/Noche)
- 🚗 **Tipo de incidente**: `CLASE_ACCIDENTE` (Choque, Atropello, Volcamiento, Caída Ocupante, Otro, Incendio)
- 🛣️ **Tipo de vía**: `DISEÑO` (Tramo de vía, Intersección, Glorieta, Puente, Ciclo Ruta, etc.)
- 📍 **Geográficas**: `COMUNA` (21 comunas y corregimientos)

**Variables descartadas:** `EXPEDIENTE`, `RADICADO`, `CBML`, `DIRECCION`, `LOCATION`, `X`, `Y`, `OBJECTID`, `BARRIO` (300+ categorías únicas) y la fecha original (ya transformada en variables temporales).

---

## 🏗️ Arquitectura del proyecto

```
Datos del incidente (hora, clase, vía, comuna)
        │
        ▼
Ingeniería de variables (periodo del día, fin de semana)
        │
        ▼
One-Hot Encoding (categorías → columnas numéricas 0/1)
        │
        ▼
Random Forest Classifier (200 árboles, profundidad 12)
        │
        ▼
Probabilidad de víctimas + Nivel de riesgo + Explicación SHAP + Recomendaciones
```

---

## 🤖 Modelo de machine learning

Se compararon tres algoritmos sobre el mismo conjunto de prueba (20%, n = 51.161):

| Modelo | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7706 | 0.7446 | 0.8406 |
| **Random Forest** ✅ | 0.7696 | 0.7399 | **0.8440** |
| XGBoost | 0.7728 | 0.7505 | 0.8469 |

**Modelo final: Random Forest**, seleccionado por:

- 🌳 Soporte directo de SHAP con `TreeExplainer`
- ⚖️ `class_weight='balanced'` compensa el desbalance 59/41%
- 🔧 Sin hiperparámetros críticos sensibles (a diferencia de XGBoost)
- ⚡ `n_jobs=-1` paraleliza el entrenamiento sobre +255K registros

**Hiperparámetros:**

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

División de datos: **80% entrenamiento / 20% prueba**, aleatoria y estratificada (`stratify=y`), validada adicionalmente con **5-fold cross-validation**.

---

## 📈 Resultados

| Métrica | Valor | Interpretación |
|---|---|---|
| **Accuracy** | 77.0% | De cada 100 incidentes, clasifica correctamente 77 |
| **F1-Score** | 0.740 | Balance entre precisión y recall |
| **ROC-AUC** | 0.844 | Buena capacidad de separación entre clases |
| **Recall (víctimas)** | 59% | Proporción de casos reales con víctimas detectados |

> El recall moderado refleja el tradeoff inherente al desbalance de clases (59% vs. 41%): subir el recall implica más falsas alarmas, bajarlo implica perder más casos reales con víctimas. El umbral actual (0.5) es ajustable según la política de riesgo de la entidad usuaria.

---

## 🧠 Interpretabilidad con SHAP

Cada predicción individual se explica con **SHAP (SHapley Additive exPlanations)** usando `TreeExplainer`:

- 🔴 **Barra roja (valor positivo)** → la variable aumentó la probabilidad de víctimas
- 🔵 **Barra azul (valor negativo)** → la variable redujo la probabilidad de víctimas
- 📏 **Longitud de la barra** → magnitud de la influencia en esa predicción puntual

**Hallazgos clave del análisis:**

| ⚠️ Aumentan la gravedad | 🛡️ Reducen la gravedad |
|---|---|
| Atropellos | Choques simples entre vehículos |
| Horario nocturno (8 p.m. – 5 a.m.) | Horario diurno (8 a.m. – 5 p.m.) |
| Intersecciones | Tramos rectos de vía |
| Caída de ocupante (motociclistas) | Lotes o predios |
| Fines de semana | Entre semana (lunes a jueves) |

---

## 📁 Estructura del repositorio

```
.
├── 01_exploracion_vial.py              # Descarga, limpieza, EDA, entrenamiento y comparación de modelos
├── app_vial.py                         # Aplicación Streamlit (6 secciones interactivas)
├── gravedad_model.pkl                  # Modelo Random Forest entrenado (generado al ejecutar 01_exploracion_vial.py)
├── feature_names_vial.pkl              # Lista de columnas/features esperadas por el modelo
├── Incidentes-Viales-Medellin.csv      # Dataset histórico (descargado automáticamente)
├── Documentacion_Tecnica_Manual_Usuario.docx   # Documentación técnica completa
└── README.md                           # Este archivo
```

---

## ⚙️ Instalación y uso

> 💡 Si solo quieres **probar la app**, no necesitas instalar nada: usa la [versión en vivo](https://predictor-vial-medellin.streamlit.app/). Los pasos siguientes son para ejecutarla localmente o modificar el código.

### 1. Clonar el repositorio

```bash
git clone https://github.com/samoyedo15/predictor-vial-medellin.git
cd predictor-vial-medellin
```

### 2. Crear entorno e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate

pip install streamlit pandas numpy scikit-learn xgboost joblib plotly shap matplotlib seaborn
```

### 3. Generar el modelo (paso obligatorio antes de usar la app)

```bash
python 01_exploracion_vial.py
```

Este script descarga el dataset, realiza la limpieza y el feature engineering, entrena y compara los tres modelos, y genera:

- `gravedad_model.pkl`
- `feature_names_vial.pkl`
- `Incidentes-Viales-Medellin.csv`

### 4. Ejecutar la aplicación

```bash
streamlit run app_vial.py
```

La aplicación se abrirá en `http://localhost:8501`.

---

## 🧭 Funcionalidades de la app

### 🔍 Predicción Individual
1. Selecciona clase de accidente, diseño de vía y comuna.
2. Ajusta hora, día de la semana y mes.
3. Presiona **"Predecir Gravedad"** para obtener:
   - Medidor de probabilidad (gauge) y nivel de riesgo (🔴/🟡/🟢)
   - Plan de prevención sugerido y priorizado
   - Gráfico SHAP con las 12 variables más influyentes en esa predicción

### 📁 Predicción por Lote
1. Descarga el CSV de ejemplo desde la app.
2. Carga tu propio archivo CSV con el formato esperado (ver abajo).
3. Ejecuta las predicciones masivas y descarga los resultados.

### 📊 Segmentación
Visualiza patrones de riesgo (hora, clase de accidente) sobre una muestra de 5.000 incidentes históricos.

### 🔬 Comparación de Modelos
Compara visualmente el desempeño de Logistic Regression, Random Forest y XGBoost.

---

## 📄 Formato de CSV para predicción por lote

El archivo debe contener las siguientes columnas:

```csv
CLASE,DISENO,COMUNA,MES,hora,dia_semana
Atropello,Interseccion,La Candelaria,9,22,4
Choque,Tramo de via,Castilla,3,8,1
Volcamiento,Glorieta,El Poblado,12,2,6
```

| Columna | Valores válidos |
|---|---|
| `CLASE` | Choque, Atropello, Volcamiento, Caida Ocupante, Otro, Incendio |
| `DISENO` | Tramo de via, Interseccion, Glorieta, Lote o Predio, Puente, Ciclo Ruta, Paso Elevado, Paso Inferior, Paso a Nivel, Tunel, Via peatonal |
| `COMUNA` | Cualquiera de las 21 comunas/corregimientos de Medellín (ver `app_vial.py`) |
| `MES` | Número de 1 a 12 |
| `hora` | Número de 0 a 23 |
| `dia_semana` | 0 = Lunes … 6 = Domingo |

---

## ⚠️ Limitaciones

- 📍 Entrenado y validado solo con datos de **Medellín**; no generaliza a otras ciudades sin reentrenamiento.
- 📅 Datos hasta **2022**; cambios de infraestructura posteriores no están capturados.
- 💀 Los fallecidos (~0.13% de los registros) se agrupan junto con heridos por su baja frecuencia.
- 🌧️ No incluye variables climáticas (lluvia, neblina, estado del pavimento).
- 🗺️ Granularidad geográfica a nivel de **comuna**, no de calle específica.
- 🚨 Es una **herramienta de apoyo**: nunca debe reemplazar el criterio experto de las autoridades de movilidad.

---

## 🔐 Ética y privacidad de los datos

- ✅ Dataset 100% público, obtenido de [datos.gov.co](https://www.datos.gov.co/Transporte/Accidentes-Viales/yu3i-jau4)
- ✅ Sin información personal (sin nombres, cédulas ni placas vehiculares)
- ✅ Fuente oficial: Secretaría de Movilidad, Alcaldía de Medellín
- ✅ Licencia abierta — Ley 1712 de 2014 (Transparencia y Acceso a la Información Pública)
- ⚠️ Las predicciones no deben usarse para estigmatizar comunas o grupos poblacionales, solo para orientar acciones de prevención vial

---

## 🛠️ Tecnologías utilizadas

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.9+ |
| Machine Learning | scikit-learn, XGBoost |
| Interpretabilidad | SHAP |
| Procesamiento de datos | pandas, NumPy |
| Visualización | Plotly, Matplotlib, Seaborn |
| Aplicación web | Streamlit |
| Serialización | joblib |

---

## 📚 Documentación adicional

Consulta el archivo [`Documentacion_Tecnica_Manual_Usuario.docx`](./Documentacion_Tecnica_Manual_Usuario.docx) incluido en este repositorio para el detalle completo de: arquitectura, datos, entrenamiento, evaluación, interpretabilidad, manual de uso paso a paso, limitaciones y consideraciones éticas.

---

## 👤 Autor

**Anderson Marin**
Curso de Profundización en Inteligencia Artificial

---

<p align="center">
  <i>Proyecto desarrollado con fines académicos a partir de datos abiertos de la Secretaría de Movilidad de Medellín.</i>
</p>
