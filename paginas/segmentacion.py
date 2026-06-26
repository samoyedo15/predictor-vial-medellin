import streamlit as st
import pandas as pd
import plotly.express as px

from utils import construir_features


def mostrar(modelo, feature_names):
    st.title("📊 Segmentación por Nivel de Riesgo")
    st.markdown("<p style='color:#9090b8;'>Muestra de 5,000 incidentes históricos segmentados por probabilidad predicha de víctimas.</p>", unsafe_allow_html=True)
    st.markdown("---")

    try:
        muestra = _cargar_muestra()
        muestra = _calcular_probabilidades(muestra, modelo, feature_names)
        _mostrar_metricas(muestra)
        _mostrar_graficas(muestra)
    except FileNotFoundError:
        st.error("No se encontró `Incidentes-Viales-Medellin.csv`. Ejecuta primero `02_entrenamiento.py`.")


def _cargar_muestra():
    df_o    = pd.read_csv("Incidentes-Viales-Medellin.csv")
    muestra = df_o.sample(n=min(5000, len(df_o)), random_state=42).copy()

    fecha_col = [c for c in muestra.columns
                 if 'FECHA' in c.upper() and 'ENCASILLADA' not in c.upper()]
    if fecha_col:
        muestra['fecha_dt']   = pd.to_datetime(muestra[fecha_col[-1]], errors='coerce')
        muestra['hora']       = muestra['fecha_dt'].dt.hour.fillna(12).astype(int)
        muestra['dia_semana'] = muestra['fecha_dt'].dt.dayofweek.fillna(2).astype(int)
    else:
        muestra['hora'] = 12
        muestra['dia_semana'] = 2

    muestra['es_fin_de_semana'] = (muestra['dia_semana'] >= 5).astype(int)
    muestra['periodo'] = muestra['hora'].apply(
        lambda h: 'Madrugada' if h < 6 else 'Mañana' if h < 12 else 'Tarde' if h < 18 else 'Noche'
    )
    return muestra


def _calcular_probabilidades(muestra, modelo, feature_names):
    muestra_dict = muestra.rename(columns={'CLASE_ACCIDENTE': 'CLASE', 'DISEÑO': 'DISENO'})

    with st.spinner("Calculando probabilidades para 5,000 incidentes…"):
        probs = [
            modelo.predict_proba(
                pd.DataFrame([construir_features(row.to_dict(), feature_names)])
            )[0][1]
            for _, row in muestra_dict.iterrows()
        ]

    muestra['Prob']  = probs
    muestra['Nivel'] = muestra['Prob'].apply(
        lambda x: 'Alto' if x >= 0.7 else ('Medio' if x >= 0.4 else 'Bajo')
    )
    return muestra


def _mostrar_metricas(muestra):
    alto  = (muestra['Nivel'] == 'Alto').sum()
    medio = (muestra['Nivel'] == 'Medio').sum()
    bajo  = (muestra['Nivel'] == 'Bajo').sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Riesgo Alto",  f"{alto}",  f"{alto/len(muestra)*100:.1f}% del total")
    c2.metric("🟡 Riesgo Medio", f"{medio}", f"{medio/len(muestra)*100:.1f}% del total")
    c3.metric("🟢 Riesgo Bajo",  f"{bajo}",  f"{bajo/len(muestra)*100:.1f}% del total")
    st.markdown("---")


def _mostrar_graficas(muestra):
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.box(
            muestra, x='Nivel', y='hora', color='Nivel',
            category_orders={'Nivel': ['Bajo', 'Medio', 'Alto']},
            color_discrete_map={'Bajo': '#2ecc71', 'Medio': '#f39c12', 'Alto': '#e74c3c'},
            title='Distribución de Hora por Nivel de Riesgo'
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), showlegend=False, height=380,
            xaxis=dict(title='Nivel de Riesgo'),
            yaxis=dict(title='Hora del día', showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if 'CLASE_ACCIDENTE' in muestra.columns:
            fig2 = px.histogram(
                muestra, x='CLASE_ACCIDENTE', color='Nivel',
                category_orders={'Nivel': ['Bajo', 'Medio', 'Alto']},
                color_discrete_map={'Bajo': '#2ecc71', 'Medio': '#f39c12', 'Alto': '#e74c3c'},
                title='Clase de Accidente por Nivel de Riesgo', barmode='group'
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), height=380,
                xaxis=dict(tickangle=25, title=''),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Cantidad')
            )
            st.plotly_chart(fig2, use_container_width=True)
