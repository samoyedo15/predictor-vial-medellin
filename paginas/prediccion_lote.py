import streamlit as st
import pandas as pd
import plotly.express as px

from utils import construir_features, clasificar_riesgo


def mostrar(modelo, feature_names):
    st.title("📁 Predicción por Lote")
    st.markdown("<p style='color:#9090b8;'>Sube un CSV con múltiples incidentes para obtener predicciones masivas y exportar los resultados.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("📋 Ver formato esperado del CSV"):
        st.code("CLASE,DISENO,COMUNA,MES,hora,dia_semana")
        st.markdown("""
        <div style='color:#9090b8; font-size:12px; line-height:1.9;'>
        ● <b style='color:white;'>CLASE</b>: Choque, Atropello, Volcamiento, Caida Ocupante, Otro, Incendio<br>
        ● <b style='color:white;'>DISENO</b>: Tramo de via, Interseccion, Glorieta, Lote o Predio, Puente…<br>
        ● <b style='color:white;'>COMUNA</b>: La Candelaria, El Poblado, Castilla…<br>
        ● <b style='color:white;'>MES</b>: número 1–12<br>
        ● <b style='color:white;'>hora</b>: 0–23<br>
        ● <b style='color:white;'>dia_semana</b>: 0 = Lunes … 6 = Domingo
        </div>
        """, unsafe_allow_html=True)

    ejemplo_csv = (
        "CLASE,DISENO,COMUNA,MES,hora,dia_semana\n"
        "Atropello,Interseccion,La Candelaria,9,22,4\n"
        "Choque,Tramo de via,Castilla,3,8,1\n"
        "Volcamiento,Glorieta,El Poblado,12,2,6\n"
        "Caida Ocupante,Tramo de via,Robledo,7,18,5\n"
        "Atropello,Paso Elevado,Guayabal,11,20,2"
    )
    st.download_button("📥 Descargar CSV de ejemplo", data=ejemplo_csv,
                       file_name="ejemplo_incidentes.csv", mime="text/csv")
    st.markdown("---")

    archivo = st.file_uploader("📂 Cargar archivo CSV", type=['csv'])
    if archivo is not None:
        df_lote = pd.read_csv(archivo)
        st.write(f"Se cargaron **{len(df_lote)} incidentes**")

        if st.button("🚀 Ejecutar predicciones", type="primary", use_container_width=True):
            resultados = _procesar_lote(df_lote, modelo, feature_names)
            _mostrar_resultados(resultados)


def _procesar_lote(df_lote, modelo, feature_names):
    resultados = []
    prog  = st.progress(0, text="Procesando incidentes…")
    total = len(df_lote)

    for i, (_, row) in enumerate(df_lote.iterrows()):
        rd = row.to_dict()
        h  = int(rd.get('hora', 12))
        ds = int(rd.get('dia_semana', 0))
        rd['es_fin_de_semana'] = 1 if ds >= 5 else 0
        rd['periodo'] = 'Madrugada' if h < 6 else 'Mañana' if h < 12 else 'Tarde' if h < 18 else 'Noche'

        d    = construir_features(rd, feature_names)
        df_r = pd.DataFrame([d])
        prob = modelo.predict_proba(df_r)[0][1]
        pred = modelo.predict(df_r)[0]

        resultados.append({
            'Clase':          rd.get('CLASE', ''),
            'Diseño':         rd.get('DISENO', ''),
            'Comuna':         rd.get('COMUNA', ''),
            'Hora':           h,
            'Prob. Víctimas': f"{prob*100:.1f}%",
            'Prob_num':       prob,
            'Riesgo':         clasificar_riesgo(prob),
            'Predicción':     '⚠️ Con víctimas' if pred == 1 else '✅ Solo daños'
        })
        prog.progress((i + 1) / total, text=f"Procesando {i+1}/{total}…")

    prog.empty()
    return resultados


def _mostrar_resultados(resultados):
    df_res = pd.DataFrame(resultados)
    alto   = len(df_res[df_res['Prob_num'] >= 0.7])
    medio  = len(df_res[(df_res['Prob_num'] >= 0.4) & (df_res['Prob_num'] < 0.7)])
    bajo   = len(df_res[df_res['Prob_num'] < 0.4])

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total procesados", len(df_res))
    c2.metric("🔴 Alto Riesgo",  alto,  f"{alto/len(df_res)*100:.1f}%")
    c3.metric("🟡 Medio Riesgo", medio, f"{medio/len(df_res)*100:.1f}%")
    c4.metric("🟢 Bajo Riesgo",  bajo,  f"{bajo/len(df_res)*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            values=[alto, medio, bajo], names=['Alto', 'Medio', 'Bajo'],
            color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71'], hole=0.55
        )
        fig_pie.update_traces(textinfo='label+percent', textfont_size=13)
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=320,
            title=dict(text='Distribución de Riesgo', font=dict(size=14, color='#9090b8')),
            legend=dict(orientation='h', y=-0.1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_hist = px.histogram(df_res, x='Prob_num', nbins=20,
                                color_discrete_sequence=['#e94560'])
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="#e74c3c",
                           annotation_text="Alto", annotation_font_color="#e74c3c")
        fig_hist.add_vline(x=0.4, line_dash="dash", line_color="#f39c12",
                           annotation_text="Medio", annotation_font_color="#f39c12")
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=320,
            title=dict(text='Distribución de Probabilidades', font=dict(size=14, color='#9090b8')),
            xaxis=dict(title='Probabilidad de víctimas', tickformat='.0%'),
            yaxis=dict(title='Cantidad', showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    df_export = df_res.sort_values('Prob_num', ascending=False).drop(columns=['Prob_num'])
    st.subheader("📋 Resultados detallados")
    st.dataframe(df_export, use_container_width=True)
    st.download_button(
        "📥 Descargar resultados CSV",
        data=df_export.to_csv(index=False),
        file_name="resultados_gravedad.csv", mime="text/csv"
    )
