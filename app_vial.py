import streamlit as st

from styles import aplicar_estilos
from utils import cargar_modelo
import paginas.dashboard as dashboard
import paginas.prediccion_individual as prediccion_individual
import paginas.prediccion_lote as prediccion_lote
import paginas.segmentacion as segmentacion
import paginas.comparacion_modelos as comparacion_modelos
import paginas.acerca_del_modelo as acerca_del_modelo

st.set_page_config(
    page_title="Seguridad Vial - Medellín",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

aplicar_estilos()

modelo, feature_names, MODEL_LOADED, MODEL_ERROR = cargar_modelo()

# ── Sidebar ──
st.sidebar.markdown("""
<div style='text-align:center; padding:15px 0 25px;'>
    <div style='font-size:46px; margin-bottom:8px;'>🚦</div>
    <div style='color:#e94560; font-weight:800; font-size:17px; margin-bottom:3px;'>Seguridad Vial</div>
    <div style='color:#7070a0; font-size:13px; letter-spacing:1.5px;'>MEDELLÍN — VALLE DE ABURRÁ</div>
    <div style='margin-top:12px; display:inline-block; padding:5px 14px;
                background:rgba(233,69,96,0.15); border-radius:20px;
                border:1px solid rgba(233,69,96,0.3);'>
        <span style='color:#e94560; font-size:13px; font-weight:700;'>● SISTEMA ACTIVO</span>
    </div>
</div>
""", unsafe_allow_html=True)

pagina = st.sidebar.radio("", [
    "🏠 Dashboard",
    "🔍 Predicción Individual",
    "📁 Predicción por Lote",
    "📊 Segmentación",
    "🔬 Comparación de Modelos",
    "ℹ️ Acerca del Modelo"
], label_visibility="collapsed")

st.sidebar.markdown("---")

with st.sidebar.expander("📖 Guía rápida"):
    st.markdown("""
    <div style='color:#9090b8; font-size:14px; line-height:2.0;'>
    <b style='color:#e94560;'>🏠 Dashboard</b><br>
    Resumen ejecutivo y hallazgos clave.<br><br>
    <b style='color:#e94560;'>🔍 Predicción Individual</b><br>
    Analiza un incidente específico.<br><br>
    <b style='color:#e94560;'>📁 Predicción por Lote</b><br>
    Sube un CSV con múltiples incidentes.<br><br>
    <b style='color:#e94560;'>📊 Segmentación</b><br>
    Explora patrones en datos históricos.<br><br>
    <b style='color:#e94560;'>🔬 Comparación</b><br>
    Los 3 modelos evaluados cara a cara.<br><br>
    <b style='color:#e94560;'>ℹ️ Acerca del Modelo</b><br>
    Metodología, datos y limitaciones.
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='text-align:center; color:#6060a0; font-size:13px; padding:14px 8px 4px;'>
    <span style='color:#9090b8;'>Desarrollado por</span><br>
    <b style='color:white; font-size:15px;'>Anderson Marin</b><br>
    <span>Curso de Profundización en IA</span><br>
    <span style='margin-top:6px; display:block; color:#505070; font-size:12px;'>Fuente: Secretaría de Movilidad<br>Alcaldía de Medellín</span>
</div>
""", unsafe_allow_html=True)

# ── Guard: modelo requerido para algunas páginas ──
PAGINAS_SIN_MODELO = {"🏠 Dashboard", "🔬 Comparación de Modelos", "ℹ️ Acerca del Modelo"}
if not MODEL_LOADED and pagina not in PAGINAS_SIN_MODELO:
    st.error(
        f"⚠️ No se encontraron los archivos del modelo.\n\n"
        f"Ejecuta primero `02_entrenamiento.py` para generar "
        f"`gravedad_model.pkl` y `feature_names_vial.pkl`.\n\n`{MODEL_ERROR}`"
    )
    st.stop()

# ── Router ──
if pagina == "🏠 Dashboard":
    dashboard.mostrar()
elif pagina == "🔍 Predicción Individual":
    prediccion_individual.mostrar(modelo, feature_names)
elif pagina == "📁 Predicción por Lote":
    prediccion_lote.mostrar(modelo, feature_names)
elif pagina == "📊 Segmentación":
    segmentacion.mostrar(modelo, feature_names)
elif pagina == "🔬 Comparación de Modelos":
    comparacion_modelos.mostrar()
elif pagina == "ℹ️ Acerca del Modelo":
    acerca_del_modelo.mostrar()
