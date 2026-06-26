import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go

from constants import CLASES, DISENOS, COMUNAS, DIAS_NOMBRE
from utils import construir_features, crear_gauge, recomendar_prevencion


def mostrar(modelo, feature_names):
    st.title("🔍 Predicción de Gravedad — Incidente Individual")
    st.markdown("<p style='color:#9090b8;'>Ingresa las características del incidente. El modelo predice en tiempo real si hubo víctimas y explica su decisión con SHAP.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🚗 Tipo de Incidente")
        clase  = st.selectbox("Clase de accidente", CLASES)
        diseno = st.selectbox("Diseño de la vía", DISENOS)
        comuna = st.selectbox("Comuna", COMUNAS)
    with col2:
        st.markdown("#### 🕐 Información Temporal")
        hora       = st.slider("Hora del incidente", 0, 23, 14)
        dia_idx    = st.selectbox("Día de la semana", DIAS_NOMBRE)
        dia_semana = DIAS_NOMBRE.index(dia_idx)
        mes        = st.slider("Mes", 1, 12, 6)
    with col3:
        st.markdown("#### 📋 Calculados Automáticamente")
        es_fds  = 1 if dia_semana >= 5 else 0
        periodo = 'Madrugada' if hora < 6 else 'Mañana' if hora < 12 else 'Tarde' if hora < 18 else 'Noche'
        st.info(f"🗓️ Fin de semana: {'Sí' if es_fds else 'No'}")
        st.info(f"🌗 Periodo: {periodo}")
        st.info(f"📍 {comuna}")

    st.markdown("---")
    incidente_raw = {
        'CLASE': clase, 'DISENO': diseno, 'COMUNA': comuna,
        'hora': hora, 'dia_semana': dia_semana,
        'es_fin_de_semana': es_fds, 'MES': mes, 'periodo': periodo
    }

    if st.button("🔍 Predecir Gravedad", type="primary", use_container_width=True):
        datos        = construir_features(incidente_raw, feature_names)
        incidente_df = pd.DataFrame([datos])
        prediccion   = modelo.predict(incidente_df)[0]
        probabilidad = modelo.predict_proba(incidente_df)[0][1]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(crear_gauge(probabilidad), use_container_width=True)
            nivel_css   = "risk-high" if probabilidad >= 0.7 else "risk-medium" if probabilidad >= 0.4 else "risk-low"
            nivel_text  = "ALTO" if probabilidad >= 0.7 else "MEDIO" if probabilidad >= 0.4 else "BAJO"
            nivel_color = "#e74c3c" if probabilidad >= 0.7 else "#f39c12" if probabilidad >= 0.4 else "#2ecc71"
            st.markdown(f"""<div class="{nivel_css}" style="text-align:center;">
                <h3 style="color:{nivel_color}; margin:0; font-size:21px;">Riesgo {nivel_text}</h3>
                <p style="color:#ccc; margin:5px 0 0; font-size:13px;">
                {'Alta probabilidad de que este incidente tenga víctimas' if probabilidad >= 0.4 else 'Probablemente solo daños materiales'}
                </p>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-card" style="margin-top:14px;">
                <p style="color:#9090b8; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Resumen del incidente evaluado</p>
                <table style="width:100%; color:#ccc; font-size:15px;">
                <tr><td style="color:#6060a0; padding:3px 0;">Clase</td><td style="text-align:right; color:white;">{clase}</td></tr>
                <tr><td style="color:#6060a0; padding:3px 0;">Vía</td><td style="text-align:right; color:white;">{diseno}</td></tr>
                <tr><td style="color:#6060a0; padding:3px 0;">Hora</td><td style="text-align:right; color:white;">{hora}:00 ({periodo})</td></tr>
                <tr><td style="color:#6060a0; padding:3px 0;">Día</td><td style="text-align:right; color:white;">{dia_idx}</td></tr>
                <tr><td style="color:#6060a0; padding:3px 0;">Mes</td><td style="text-align:right; color:white;">{mes}</td></tr>
                <tr><td style="color:#6060a0; padding:3px 0;">Fin de semana</td><td style="text-align:right; color:white;">{'Sí' if es_fds else 'No'}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### 🛡️ Plan de Prevención")
            acciones = recomendar_prevencion(incidente_raw, probabilidad)
            for icon, accion, prioridad in acciones:
                p_color = "#e74c3c" if prioridad == "Alto" else "#f39c12" if prioridad == "Medio" else "#2ecc71"
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border-radius:10px;
                            padding:12px 15px; margin:8px 0; border-left:3px solid {p_color};">
                    <span style="font-size:18px;">{icon}</span>
                    <span style="color:#e0e0e0; margin-left:10px; font-size:15px;">{accion}</span>
                    <br><span style="color:{p_color}; font-size:13px; font-weight:700; margin-left:32px;">● Prioridad {prioridad}</span>
                </div>""", unsafe_allow_html=True)

        _mostrar_shap(modelo, incidente_df)


def _mostrar_shap(modelo, incidente_df):
    st.markdown("---")
    st.header("🧠 Explicación SHAP — ¿Por qué este resultado?")
    st.markdown("""<p style='color:#9090b8; font-size:13px;'>
    SHAP descompone la predicción en la contribución de cada variable.
    <b style='color:#e74c3c;'>Rojo</b> = aumentó la probabilidad de víctimas.
    <b style='color:#3498db;'>Azul</b> = la redujo.
    La longitud de la barra indica cuánto influyó.
    </p>""", unsafe_allow_html=True)

    explainer   = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(incidente_df)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]
        if shap_vals.ndim == 2:
            shap_vals = shap_vals[:, 1]

    shap_df = pd.DataFrame({
        'Variable': list(incidente_df.columns),
        'SHAP':     shap_vals,
        'Valor':    incidente_df.iloc[0].values
    })
    shap_df['Abs'] = shap_df['SHAP'].abs()
    shap_df = shap_df.nlargest(12, 'Abs').sort_values('SHAP')

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=shap_df['SHAP'], y=shap_df['Variable'], orientation='h',
        marker_color=shap_df['SHAP'].apply(lambda v: '#e74c3c' if v > 0 else '#3498db'),
        text=shap_df['SHAP'].apply(lambda v: f'{v:+.3f}'), textposition='outside',
        textfont=dict(color='white', size=11)
    ))
    fig_shap.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12), height=440,
        title=dict(text='Top 12 variables que influyeron en esta predicción',
                   font=dict(size=13, color='#9090b8')),
        xaxis=dict(title='Impacto SHAP (unidades de probabilidad log-odds)',
                   showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   zeroline=True, zerolinecolor='rgba(255,255,255,0.25)'),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=90, t=50, b=40)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="info-card" style="border-left-color:#e74c3c;">
        <p style="color:#e74c3c; margin:0; font-weight:700; font-size:15px;">🔴 Barras rojas</p>
        <p style="color:#ccc; margin:5px 0 0; font-size:15px;">Esta variable <b>aumentó</b> la probabilidad de víctimas para este incidente específico</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card" style="border-left-color:#3498db;">
        <p style="color:#3498db; margin:0; font-weight:700; font-size:15px;">🔵 Barras azules</p>
        <p style="color:#ccc; margin:5px 0 0; font-size:15px;">Esta variable <b>redujo</b> la probabilidad de víctimas para este incidente específico</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card">
        <p style="color:#e94560; margin:0; font-weight:700; font-size:15px;">📏 Longitud de barra</p>
        <p style="color:#ccc; margin:5px 0 0; font-size:15px;">Barra más larga = mayor influencia en la decisión del modelo para este caso</p>
        </div>""", unsafe_allow_html=True)
