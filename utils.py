import joblib
import pandas as pd
import plotly.graph_objects as go


def cargar_modelo():
    try:
        modelo = joblib.load('gravedad_model.pkl')
        feature_names = joblib.load('feature_names_vial.pkl')
        return modelo, feature_names, True, ""
    except FileNotFoundError as e:
        return None, None, False, str(e)


def construir_features(row, feature_names):
    datos = {col: 0 for col in feature_names}
    for campo in ['hora', 'dia_semana', 'es_fin_de_semana', 'MES']:
        if campo in datos:
            datos[campo] = float(row.get(campo, 0))
    mapeo_prefijos = {
        'CLASE':   'CLASE_ACCIDENTE',
        'DISENO':  'DISEÑO',
        'COMUNA':  'COMUNA',
        'periodo': 'periodo',
    }
    for campo_form, prefijo_real in mapeo_prefijos.items():
        valor = str(row.get(campo_form, ''))
        if not valor:
            continue
        col_name = f"{prefijo_real}_{valor}"
        if col_name in datos:
            datos[col_name] = 1
    return datos


def clasificar_riesgo(p):
    if p >= 0.7:
        return "🔴 Alto"
    elif p >= 0.4:
        return "🟡 Medio"
    else:
        return "🟢 Bajo"


def crear_gauge(probabilidad):
    color = "#2ecc71" if probabilidad < 0.4 else "#f39c12" if probabilidad < 0.7 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad * 100,
        number={'suffix': '%', 'font': {'size': 50, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'rgba(255,255,255,0.4)',
                     'tickfont': {'color': 'rgba(255,255,255,0.6)'}, 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.78},
            'bgcolor': 'rgba(15,15,35,0.8)',
            'borderwidth': 1, 'bordercolor': 'rgba(255,255,255,0.08)',
            'steps': [
                {'range': [0, 40],   'color': 'rgba(46,204,113,0.10)'},
                {'range': [40, 70],  'color': 'rgba(243,156,18,0.10)'},
                {'range': [70, 100], 'color': 'rgba(231,76,60,0.10)'}
            ],
            'threshold': {'line': {'color': 'white', 'width': 2},
                          'thickness': 0.82, 'value': probabilidad * 100}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}, height=300, margin=dict(l=30, r=30, t=40, b=20)
    )
    return fig


def recomendar_prevencion(row, prob):
    if prob < 0.4:
        return [("✅", "Zona de riesgo controlado — mantener monitoreo regular de patrones en la vía", "Bajo")]
    acciones = []
    clase   = row.get('CLASE', '')
    diseno  = row.get('DISENO', '')
    periodo = row.get('periodo', '')
    if clase == 'Atropello':
        acciones.append(("🚶", "Reforzar señalización peatonal y pasos de cebra en la zona", "Alto"))
    if periodo in ['Noche', 'Madrugada']:
        acciones.append(("💡", "Evaluar iluminación vial — incidentes nocturnos tienen mayor probabilidad de víctimas", "Alto"))
    if diseno == 'Interseccion':
        acciones.append(("🚦", "Revisar semáforos, tiempos de ciclo y reductores de velocidad en la intersección", "Alto"))
    if clase == 'Volcamiento':
        acciones.append(("🛣️", "Inspeccionar estado de la vía, señalización de curvas y velocidad máxima permitida", "Medio"))
    if int(row.get('es_fin_de_semana', 0)) == 1:
        acciones.append(("👮", "Aumentar presencia de agentes de tránsito los fines de semana", "Medio"))
    if diseno in ['Ciclo Ruta', 'Via peatonal']:
        acciones.append(("🚴", "Evaluar separación física entre ciclistas/peatones y vehículos motorizados", "Medio"))
    if clase == 'Caida Ocupante':
        acciones.append(("🏍️", "Campaña de uso obligatorio de casco y elementos de protección personal", "Medio"))
    if not acciones:
        acciones.append(("📋", "Registrar incidente y analizar patrones históricos acumulados de la zona", "Bajo"))
    return acciones
