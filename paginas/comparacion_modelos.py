import streamlit as st
import pandas as pd
import plotly.graph_objects as go


_DATOS_COMPARACION = {
    'Modelo':   ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [0.7706, 0.7696, 0.7728],
    'F1-Score': [0.7446, 0.7399, 0.7505],
    'ROC-AUC':  [0.8406, 0.8440, 0.8469]
}


def mostrar():
    st.title("🔬 Comparación de Modelos")
    st.markdown("<p style='color:#9090b8;'>Se evaluaron 3 algoritmos con el mismo conjunto de prueba (20%, n=51,161 incidentes).</p>", unsafe_allow_html=True)
    st.markdown("---")

    comp = pd.DataFrame(_DATOS_COMPARACION)
    st.caption("Métricas obtenidas corriendo `02_entrenamiento.py` — conjunto de prueba 20%")

    _grafico_comparativo(comp)
    st.markdown("---")
    _tabla_comparativa(comp)
    st.markdown("---")
    _justificacion_random_forest()


def _grafico_comparativo(comp):
    fig = go.Figure()
    colores = ['#3498db', '#e94560', '#f39c12']
    for i, metrica in enumerate(['Accuracy', 'F1-Score', 'ROC-AUC']):
        fig.add_trace(go.Bar(
            name=metrica, x=comp['Modelo'], y=comp[metrica],
            text=comp[metrica].apply(lambda x: f'{x:.3f}'), textposition='outside',
            textfont=dict(color='white', size=13),
            marker_color=colores[i]
        ))
    fig.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), height=420,
        yaxis=dict(range=[0.6, 1.0], showgrid=True,
                   gridcolor='rgba(255,255,255,0.05)', title='Score'),
        legend=dict(orientation='h', y=1.12),
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


def _tabla_comparativa(comp):
    st.subheader("📋 Tabla Comparativa")
    for _, row in comp.iterrows():
        es_mejor = row['Modelo'] == 'Random Forest'
        borde    = "border:2px solid rgba(233,69,96,0.45);" if es_mejor else "border:1px solid rgba(255,255,255,0.07);"
        badge    = "<span style='background:#e94560; color:white; font-size:10px; padding:2px 8px; border-radius:10px; margin-left:8px;'>SELECCIONADO</span>" if es_mejor else ""
        st.markdown(f"""
        <div class="glass-card" style="{borde} margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div><span style="color:white; font-weight:700; font-size:15px;">{row['Modelo']}</span>{badge}</div>
                <div style="display:flex; gap:28px;">
                    <div style="text-align:center;">
                        <div style="color:#7070a0; font-size:13px;">Accuracy</div>
                        <div style="color:white; font-weight:800; font-size:20px;">{row['Accuracy']:.3f}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#7070a0; font-size:13px;">F1-Score</div>
                        <div style="color:white; font-weight:800; font-size:20px;">{row['F1-Score']:.3f}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#7070a0; font-size:13px;">ROC-AUC</div>
                        <div style="color:#e94560; font-weight:800; font-size:20px;">{row['ROC-AUC']:.3f}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _justificacion_random_forest():
    st.header("✅ ¿Por qué Random Forest?")
    st.markdown("""<div class="risk-low">
    <h4 style="color:#2ecc71; margin-top:0; font-size:15px;">Justificación de la selección</h4>
    <div style="color:#ccc; font-size:15px; line-height:2.0;">
    XGBoost obtuvo el mejor ROC-AUC (0.847 vs 0.844), pero la diferencia es marginal
    y no cambia las decisiones prácticas. Random Forest fue elegido por 4 razones:<br><br>
    🌳 <b>SHAP con TreeExplainer</b> — interpretación directa en probabilidad, no log-odds<br>
    ⚖️ <b>class_weight='balanced'</b> — compensa el desbalance 59/41 sin ajustes finos<br>
    🔧 <b>Sin hiperparámetros críticos</b> — no hay sensibilidad a learning_rate ni gamma<br>
    ⚡ <b>n_jobs=-1</b> — paraleliza sobre los 256K registros eficientemente
    </div></div>""", unsafe_allow_html=True)
