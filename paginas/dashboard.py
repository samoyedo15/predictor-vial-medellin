import streamlit as st
import plotly.graph_objects as go


def mostrar():
    st.markdown("""
    <div style='text-align:center; padding:10px 0 4px;'>
        <span style='color:#7070a0; font-size:13px; letter-spacing:2px; text-transform:uppercase;'>
        Sistema de Inteligencia Artificial para Seguridad Vial
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.title("🚦 Predicción de Gravedad — Incidentes Viales")
    st.markdown("""
    <p style='color:#9090b8; font-size:15px; text-align:center; max-width:720px; margin:0 auto 24px; line-height:1.8;'>
    Modelo de machine learning entrenado con
    <b style='color:white;'>255,804 incidentes reales</b> de Medellín (2014–2022) para predecir si un
    accidente resultará en <b style='color:#e74c3c;'>víctimas (heridos/fallecidos)</b>
    o <b style='color:#2ecc71;'>solo daños materiales</b>.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.header("📈 Rendimiento del Modelo")
    st.markdown("<p style='color:#9090b8; font-size:13px; margin-top:-8px;'>¿Qué tan bien predice? Estas 4 métricas resumen la calidad del modelo.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-explain">
        <div class="mt">Accuracy</div><div class="mv">77.0%</div>
        <div class="md">De cada 100 incidentes, el modelo clasifica correctamente 77. Considera ambas clases juntas.</div>
        <div class="bar-bg"><div class="bar-fill" style="width:77%;"></div></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-explain">
        <div class="mt">F1-Score</div><div class="mv">0.740</div>
        <div class="md">Balance entre precisión y recall. Clave cuando las clases están desbalanceadas (59 vs 41%).</div>
        <div class="bar-bg"><div class="bar-fill" style="width:74%;"></div></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-explain">
        <div class="mt">ROC-AUC</div><div class="mv">0.844</div>
        <div class="md">Capacidad de separar ambas clases. 0.5 = aleatorio, 1.0 = perfecto. 0.84 es considerado bueno.</div>
        <div class="bar-bg"><div class="bar-fill" style="width:84%;"></div></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-explain">
        <div class="mt">Recall Víctimas</div><div class="mv">59%</div>
        <div class="md">Del total de incidentes reales con víctimas, el modelo detecta el 59%. Mide cuántos "no se escapan".</div>
        <div class="bar-bg"><div class="bar-fill" style="width:59%;"></div></div>
        </div>""", unsafe_allow_html=True)

    with st.expander("🤔 ¿Por qué el Recall de víctimas es solo 59%? — Haz clic para entender el tradeoff"):
        st.markdown("""
        <div style='color:#b0b0c8; font-size:15px; line-height:1.9;'>
        El dataset tiene <b style='color:white;'>59% incidentes con víctimas</b> vs 41% solo con daños.
        Este desbalance crea un tradeoff inevitable:<br><br>
        ● <b style='color:#e74c3c;'>Subir el recall</b> → el modelo alerta más, pero genera más falsas alarmas<br>
        ● <b style='color:#2ecc71;'>Bajar el recall</b> → menos falsas alarmas, pero se pierden más casos reales con víctimas<br><br>
        El umbral actual (0.5) es un punto de equilibrio razonable. En producción se puede ajustar
        según la política de riesgo aceptable de la entidad que lo use.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.header("📊 El Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Incidentes", "255,804")
    c2.metric("Con Víctimas", "59.2%")
    c3.metric("Solo Daños", "40.8%")
    c4.metric("Años cubiertos", "2014 – 2022")

    st.markdown("")
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(
            x=['Con Víctimas<br>(heridos + fallecidos)', 'Solo Daños<br>Materiales'],
            y=[59.2, 40.8],
            marker_color=['#e74c3c', '#2ecc71'],
            text=['59.2%', '40.8%'], textposition='outside',
            textfont=dict(color='white', size=16, family='Arial Black'),
            width=0.45
        ))
        fig_d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=270,
            title=dict(text='Distribución del Target (Variable Objetivo)', font=dict(size=13, color='#9090b8')),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 75], ticksuffix='%'),
            xaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_d, use_container_width=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="min-height:220px;">
            <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:14px;">VARIABLES QUE USA EL MODELO</p>
            <div style="color:#b0b0c8; font-size:15px; line-height:2.1;">
            🕐 Hora del incidente<br>
            📅 Día de la semana<br>
            🗓️ ¿Fin de semana?<br>
            🌗 Periodo del día<br>
            📆 Mes del año<br>
            🚗 Clase de accidente<br>
            🛣️ Diseño de la vía<br>
            📍 Comuna
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.header("⚙️ ¿Cómo Funciona el Sistema?")
    cols = st.columns(5)
    pasos = [
        ("1", "📥", "Datos del Incidente", "Hora, tipo, ubicación y día"),
        ("2", "🔧", "Feature Engineering", "Crea 'periodo del día' y 'es fin de semana'"),
        ("3", "🧮", "One-Hot Encoding", "Categorías → columnas numéricas (0/1)"),
        ("4", "🌳", "Random Forest", "200 árboles de decisión votan"),
        ("5", "📊", "Resultado", "Probabilidad + riesgo + recomendaciones SHAP"),
    ]
    for col, (num, icon, titulo, desc) in zip(cols, pasos):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div style='font-size:26px; margin:8px 0;'>{icon}</div>
                <div style='color:white; font-weight:700; font-size:15px; margin-bottom:8px;'>{titulo}</div>
                <div style='color:#8080b0; font-size:13px; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.header("💡 Hallazgos Clave del Análisis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="risk-high">
        <h4 style="color:#e74c3c; margin-top:0; font-size:15px;">⚠️ Factores que Aumentan la Gravedad</h4>
        <div style="color:#ddd; font-size:15px; line-height:2.1;">
        🚶 <b>Atropellos</b> — peatones son los más vulnerables<br>
        🌙 <b>Horario nocturno</b> (8 PM – 5 AM) — menor visibilidad y reflejos<br>
        🚦 <b>Intersecciones</b> — velocidades cruzadas y puntos de conflicto<br>
        🏍️ <b>Caída de ocupante</b> — mayormente motociclistas sin protección<br>
        🎉 <b>Fines de semana</b> — mayor incidencia de alcohol y fatiga
        </div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="risk-low">
        <h4 style="color:#2ecc71; margin-top:0; font-size:15px;">🛡️ Factores que Reducen la Gravedad</h4>
        <div style="color:#ddd; font-size:15px; line-height:2.1;">
        🚗 <b>Choques simples</b> entre vehículos — estructura protege<br>
        ☀️ <b>Horario diurno</b> (8 AM – 5 PM) — mejor visibilidad y atención<br>
        🛣️ <b>Tramos rectos</b> — menos conflictos de trayectorias<br>
        🏢 <b>Lotes o predios</b> — velocidades bajas al maniobrar<br>
        📅 <b>Entre semana</b> (lunes – jueves) — conducción más rutinaria
        </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:20px; background:linear-gradient(135deg,rgba(233,69,96,0.08),rgba(194,49,82,0.08));
                border-radius:16px; border:1px solid rgba(233,69,96,0.18);'>
        <p style='color:white; font-size:17px; font-weight:700; margin:0 0 6px;'>¿Listo para analizar un incidente?</p>
        <p style='color:#9090b8; font-size:13px; margin:0;'>
            Navega a <b style='color:#e94560;'>🔍 Predicción Individual</b> en el menú lateral
        </p>
    </div>
    """, unsafe_allow_html=True)
