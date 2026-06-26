import streamlit as st


def mostrar():
    st.title("ℹ️ Acerca del Modelo")
    st.markdown("---")

    _seccion_que_predice()
    st.markdown("---")
    _seccion_datos()
    st.markdown("---")
    _seccion_variables()
    st.markdown("---")
    _seccion_arquitectura()
    st.markdown("---")
    _seccion_shap()
    st.markdown("---")
    _seccion_limitaciones_etica()
    st.markdown("---")
    _pie_de_pagina()


def _seccion_que_predice():
    st.header("🎯 ¿Qué Predice Este Modelo?")
    st.markdown("""<div class="info-card">
    <p style="color:#ddd; font-size:15px; line-height:1.9; margin:0;">
    Este sistema usa <b style="color:#e94560;">Random Forest</b> entrenado con
    <b style="color:white;">255,804 incidentes viales reales</b> de la Secretaría de Movilidad de Medellín
    (2014–2022) para predecir si un accidente resultará en
    <b style="color:#e74c3c;">víctimas (heridos o fallecidos)</b> o
    <b style="color:#2ecc71;">solo daños materiales</b>.<br><br>
    El objetivo <b>no es predecir el futuro</b>, sino estimar la gravedad probable dado el contexto del incidente
    (hora, tipo, vía, lugar) para apoyar decisiones de prevención y asignación de recursos.
    Cada predicción incluye una explicación SHAP que muestra <b style="color:white;">por qué el modelo llegó a ese resultado</b>.
    </p></div>""", unsafe_allow_html=True)


def _seccion_datos():
    st.header("📊 Los Datos")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
        <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:14px;">FUENTE Y COBERTURA</p>
        <table style="width:100%; color:#ccc; font-size:15px; line-height:2.2; border-collapse:collapse;">
        <tr><td style="color:#6060a0;">Fuente</td><td>Secretaría de Movilidad — Alcaldía de Medellín</td></tr>
        <tr><td style="color:#6060a0;">Portal</td><td>datos.gov.co (datos abiertos)</td></tr>
        <tr><td style="color:#6060a0;">Periodo</td><td>2014 – 2022 (8 años)</td></tr>
        <tr><td style="color:#6060a0;">Registros</td><td>255,804 incidentes</td></tr>
        <tr><td style="color:#6060a0;">Licencia</td><td>Datos abiertos — Ley 1712/2014</td></tr>
        <tr><td style="color:#6060a0;">Info personal</td><td>Ninguna (sin nombres ni placas)</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
        <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:14px;">DISTRIBUCIÓN DEL TARGET</p>
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; color:#ccc; font-size:13px; margin-bottom:4px;">
                <span>⚠️ Con víctimas (heridos + fallecidos)</span>
                <span style="color:#e74c3c; font-weight:700;">59.2%</span>
            </div>
            <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:7px;">
                <div style="width:59.2%; height:100%; background:#e74c3c; border-radius:4px;"></div>
            </div>
        </div>
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; color:#ccc; font-size:13px; margin-bottom:4px;">
                <span>✅ Solo daños materiales</span>
                <span style="color:#2ecc71; font-weight:700;">40.8%</span>
            </div>
            <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:7px;">
                <div style="width:40.8%; height:100%; background:#2ecc71; border-radius:4px;"></div>
            </div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; color:#ccc; font-size:13px; margin-bottom:4px;">
                <span>💀 Fallecidos (subconjunto de víctimas)</span>
                <span style="color:#f39c12; font-weight:700;">~0.13%</span>
            </div>
            <div style="background:rgba(255,255,255,0.07); border-radius:4px; height:7px;">
                <div style="width:1%; height:100%; background:#f39c12; border-radius:4px;"></div>
            </div>
        </div>
        <p style="color:#505070; font-size:13px; margin-top:12px;">Los ~326 fallecidos se agrupan con heridos por su baja frecuencia relativa</p>
        </div>
        """, unsafe_allow_html=True)


def _seccion_variables():
    st.header("🔧 Variables del Modelo (Feature Engineering)")
    st.markdown("<p style='color:#9090b8; font-size:13px;'>El modelo no usa dirección ni expediente — solo información contextual generalizable a cualquier incidente.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card">
        <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:12px;">⏰ TEMPORALES</p>
        <div style="color:#ccc; font-size:15px; line-height:2.3;">
        <b style="color:white;">hora</b> — 0 a 23<br>
        <b style="color:white;">dia_semana</b> — 0=Lun … 6=Dom<br>
        <b style="color:white;">es_fin_de_semana</b> — 0/1<br>
        <b style="color:white;">MES</b> — 1 a 12<br>
        <b style="color:white;">periodo</b> — Madrugada / Mañana / Tarde / Noche
        </div>
        <p style="color:#505070; font-size:13px; margin-top:10px;">Extraídas de la columna FECHA del CSV original</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
        <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:12px;">🚗 TIPO DE INCIDENTE</p>
        <div style="color:#ccc; font-size:15px; line-height:2.3;">
        <b style="color:white;">CLASE_ACCIDENTE</b><br>
        Choque · Atropello · Volcamiento<br>
        Caida Ocupante · Otro · Incendio<br><br>
        <b style="color:white;">DISEÑO</b> (tipo de vía)<br>
        Intersección · Tramo · Glorieta<br>
        Puente · Ciclo Ruta · …
        </div>
        <p style="color:#505070; font-size:13px; margin-top:10px;">Codificadas con One-Hot Encoding (drop_first=True para evitar multicolinealidad)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
        <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:12px;">📍 GEOGRÁFICAS</p>
        <div style="color:#ccc; font-size:15px; line-height:2.3;">
        <b style="color:white;">COMUNA</b><br>
        21 comunas y corregimientos<br>
        de Medellín y el Valle de Aburrá<br><br>
        <b style="color:#505070;">Excluidas:</b><br>
        <span style="color:#404060;">BARRIO — 300+ categorías únicas</span><br>
        <span style="color:#404060;">DIRECCION — texto libre</span>
        </div>
        <p style="color:#505070; font-size:13px; margin-top:10px;">BARRIO excluido: demasiadas categorías harían el modelo frágil</p>
        </div>
        """, unsafe_allow_html=True)


def _seccion_arquitectura():
    st.header("🌳 Arquitectura y Validación")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="glass-card">
        <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:14px;">HIPERPARÁMETROS — RANDOM FOREST</p>
        <table style="width:100%; color:#ccc; font-size:15px; line-height:2.4; border-collapse:collapse;">
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="color:#7070a0; padding:2px 0; width:40%;">n_estimators</td>
            <td><b style="color:white;">200 árboles</b></td>
            <td style="color:#505070; font-size:14px;">Estabilidad en predicciones</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="color:#7070a0; padding:2px 0;">max_depth</td>
            <td><b style="color:white;">12 niveles</b></td>
            <td style="color:#505070; font-size:14px;">Evita sobreajuste severo</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="color:#7070a0; padding:2px 0;">class_weight</td>
            <td><b style="color:white;">'balanced'</b></td>
            <td style="color:#505070; font-size:14px;">Compensa el 59/41% desbalance</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="color:#7070a0; padding:2px 0;">random_state</td>
            <td><b style="color:white;">42</b></td>
            <td style="color:#505070; font-size:14px;">Reproducibilidad garantizada</td>
        </tr>
        <tr>
            <td style="color:#7070a0; padding:2px 0;">n_jobs</td>
            <td><b style="color:white;">-1</b></td>
            <td style="color:#505070; font-size:14px;">Usa todos los núcleos del CPU</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
        <p style="color:#e94560; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:14px;">VALIDACIÓN</p>
        <div style="color:#ccc; font-size:15px; line-height:2.1;">
        <b style="color:white;">Split 80/20 estratificado</b><br>
        Entrenamiento: ~204,643 reg.<br>
        Prueba: ~51,161 reg.<br><br>
        <b style="color:white;">Cross-Validation: 5 folds</b><br>
        Sobre el conjunto de entrenamiento<br><br>
        <span style="color:#505070; font-size:14px;">Split aleatorio (no temporal)<br>
        porque las features no dependen<br>del orden histórico.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)


def _seccion_shap():
    st.header("🧠 Interpretabilidad con SHAP")
    st.markdown("""
    <div class="info-card">
    <p style="color:#ddd; font-size:15px; line-height:1.9; margin:0;">
    <b style="color:#e94560;">SHAP</b> (SHapley Additive exPlanations) es un método basado en teoría de juegos cooperativos que
    descompone la predicción de cada incidente en la contribución <i>individual</i> de cada variable.<br><br>
    A diferencia de la importancia global del Random Forest, SHAP responde:
    <b style="color:white;">"¿Por qué el modelo predijo THIS para ESTE incidente específico?"</b><br><br>
    ● Valor SHAP <b style="color:#e74c3c;">positivo (rojo)</b> → esa variable empujó la predicción hacia "Con víctimas"<br>
    ● Valor SHAP <b style="color:#3498db;">negativo (azul)</b> → esa variable empujó hacia "Solo daños"<br>
    ● La magnitud indica cuánto influyó: barra larga = variable decisiva para este caso
    </p>
    </div>
    """, unsafe_allow_html=True)


def _seccion_limitaciones_etica():
    col1, col2 = st.columns(2)
    with col1:
        st.header("⚠️ Limitaciones")
        st.markdown("""<div class="risk-medium">
        <div style="color:#ddd; font-size:15px; line-height:2.2;">
        📍 <b>Solo Medellín</b> — no generaliza a otras ciudades sin reentrenamiento<br>
        📅 <b>Datos hasta 2022</b> — cambios en infraestructura post-2022 no capturados<br>
        💀 <b>Fallecidos agrupados</b> — no distingue heridos leves de fatales (~0.13%)<br>
        🌧️ <b>Sin clima</b> — lluvia, neblina, pavimento húmedo no están incluidos<br>
        🗺️ <b>Solo a nivel de comuna</b> — no ubica calles específicas de alto riesgo<br>
        🚨 <b>Herramienta de apoyo</b> — nunca debe reemplazar criterio experto
        </div></div>""", unsafe_allow_html=True)
    with col2:
        st.header("📜 Ética y Datos")
        st.markdown("""<div class="risk-low">
        <div style="color:#ddd; font-size:15px; line-height:2.2;">
        🔓 <b>Datos 100% públicos</b> — datos.gov.co, acceso abierto sin restricciones<br>
        👤 <b>Sin información personal</b> — ningún nombre, cédula ni matrícula<br>
        🏛️ <b>Fuente oficial</b> — Secretaría de Movilidad, Alcaldía de Medellín<br>
        ⚖️ <b>Licencia abierta</b> — Ley 1712/2014 (Transparencia y acceso a info)<br>
        🎓 <b>Uso académico</b> — Curso de Profundización en IA<br>
        🤝 <b>Sin sesgo de selección</b> — todos los incidentes reportados son incluidos
        </div></div>""", unsafe_allow_html=True)


def _pie_de_pagina():
    st.markdown("""
    <div style="text-align:center; color:#505070; padding:18px; background:rgba(255,255,255,0.02); border-radius:14px;">
        <p style="margin:0; font-size:14px;">Desarrollado por <b style="color:white;">Anderson Marin</b> — Curso de Profundización en IA</p>
        <p style="margin:6px 0 0; font-size:12px;">
            Dataset: Accidentes Viales —
            <a href="https://www.datos.gov.co/Transporte/Accidentes-Viales/yu3i-jau4" style="color:#e94560;">datos.gov.co</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
