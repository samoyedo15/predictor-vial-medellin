"""
Módulo compartido de carga y preprocesamiento de datos.
Importado por los scripts 01_exploracion.py, 02_entrenamiento.py, etc.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RUTA_CSV = "Incidentes-Viales-Medellin.csv"


def cargar_datos(ruta=RUTA_CSV):
    """Carga el CSV y corrige problemas de encoding y categorías duplicadas."""
    df = pd.read_csv(ruta)

    df['GRAVEDAD_ACCIDENTE'] = df['GRAVEDAD_ACCIDENTE'].replace({
        'Solo da\\xF1os': 'Solo daños',
        'Solo danos':     'Solo daños',
    })
    df['DISEÑO'] = df['DISEÑO'].replace({'Pont\\xF3n': 'Pontón'})
    df['CLASE_ACCIDENTE'] = df['CLASE_ACCIDENTE'].str.strip().replace({
        'Caída de Ocupante': 'Caida Ocupante',
        'Caída Ocupante':    'Caida Ocupante',
        'Caida de Ocupante': 'Caida Ocupante',
    })
    return df


def crear_features_temporales(df):
    """Extrae hora, día de la semana, fin de semana y periodo del día desde la fecha."""
    df = df.copy()

    fecha_col = None
    for col in df.columns:
        if 'FECHA' in col.upper() and 'ENCASILLADA' not in col.upper():
            try:
                test = pd.to_datetime(df[col].head(10), errors='coerce')
                if test.notna().sum() > 0:
                    if fecha_col is None or df[col].str.len().mean() > df[fecha_col].str.len().mean():
                        fecha_col = col
            except Exception:
                pass

    if fecha_col is None:
        raise ValueError("No se encontró columna de fecha en el dataset.")

    df['fecha_dt']          = pd.to_datetime(df[fecha_col], errors='coerce')
    df['hora']              = df['fecha_dt'].dt.hour
    df['dia_semana']        = df['fecha_dt'].dt.dayofweek
    df['es_fin_de_semana']  = (df['dia_semana'] >= 5).astype(int)
    df['periodo']           = df['hora'].apply(_clasificar_periodo)
    return df


def _clasificar_periodo(hora):
    if pd.isna(hora):
        return 'Desconocido'
    hora = int(hora)
    if hora < 6:
        return 'Madrugada'
    elif hora < 12:
        return 'Mañana'
    elif hora < 18:
        return 'Tarde'
    else:
        return 'Noche'


def limpiar_datos(df):
    """Crea variable objetivo binaria, elimina columnas innecesarias y filas con nulos."""
    df = df.copy()

    df['victimas'] = df['GRAVEDAD_ACCIDENTE'].apply(
        lambda x: 1 if x in ['Con heridos', 'Con muertos'] else 0
    )

    cols_eliminar = []
    for col in df.columns:
        col_upper = col.upper()
        if any(x in col_upper for x in ['EXPEDIENTE', 'RADICADO', 'CBML',
                                          'LOCATION', 'DIRECCION', 'OBJECTID',
                                          'FECHA', 'GRAVEDAD_ACCIDENTE', 'NUMCOMUNA']):
            cols_eliminar.append(col)
        if col_upper in ['X', 'Y', 'AÑO', 'ANO', 'BARRIO']:
            cols_eliminar.append(col)

    if 'fecha_dt' in df.columns:
        cols_eliminar.append('fecha_dt')

    cols_protegidas = {'victimas', 'hora', 'dia_semana', 'es_fin_de_semana', 'periodo'}
    cols_eliminar = [c for c in cols_eliminar if c in df.columns and c not in cols_protegidas]
    df = df.drop(columns=cols_eliminar)

    df = df.dropna(subset=['CLASE_ACCIDENTE', 'DISEÑO', 'COMUNA', 'hora'])
    df = df[~df['COMUNA'].isin(['0', 'In', 'AU', 'SN'])]
    df['CLASE_ACCIDENTE'] = df['CLASE_ACCIDENTE'].str.strip().replace({
        'Caída de Ocupante': 'Caida Ocupante',
        'Caída Ocupante':    'Caida Ocupante',
    })
    return df


def codificar_y_dividir(df, test_size=0.2, random_state=42):
    """Aplica One-Hot Encoding y divide en train/test estratificado."""
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop('victimas', axis=1)
    y = df['victimas']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns)


def pipeline_completo(ruta=RUTA_CSV):
    """Ejecuta todo el pipeline: carga → features → limpieza → encoding → split."""
    df = cargar_datos(ruta)
    df = crear_features_temporales(df)
    df = limpiar_datos(df)
    X_train, X_test, y_train, y_test, feature_names = codificar_y_dividir(df)
    return X_train, X_test, y_train, y_test, feature_names
