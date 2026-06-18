
#==========================================================#
# E S T R U C T U R A   G E N E R A L  D E   L A    A P P


# - Carga de Libreiras

# - Asignacion de rutas y constantes

# - Funciones de Validacion y carga


# Funciones de Preprocesamiento 

# Funciones relacionados de los Moedelos

# Modulos de las páginas requeridas
        # inicio
        # exploracion
        # preprocesamiento
        # modelado
        # pca
        # simulador
        # imagenes
        # procesamiento de imagenes
        # conclusiones

# Funcion Principal
#  ejecuta lo necesario de las anteriores y el sidebar que 
#  que te permite elegir que modulo y por ende que página se ve


#=============================================================#


## ===========================================================##

#  L O G I C A

   # Se usan los distintos modulos como si fuera un rompecabezas

# El modulo principal, carga un sidebar dónde hay distintas
# Opciones que ejecutan, las distintas funciones que llaman lo que se 
# debería ver en pantalla.




## ===========================================================##


## Librerias para la carga de archivos y navegacion de los mismos
import json
import sys
from pathlib import Path



## Librerias para el funcionameintono 
import numpy as np
import pandas as pd


import altair as alt  # Librerias de visualización
import plotly.express as px

import cv2    # Libreria de Computer Vision del libro.
import joblib # Carga de modelos 


import streamlit as st # Para hacer el app.


## Al no ser parte de los modelos de clasificación y de regresión, se hace el PCA aquí
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix



## Variables donde se almacenan las direcciónes de los archivos.

BASE_DIR = Path(__file__).parent #  La parte larga que empieza desde el disco C:
                                #   donde esta el archivo actual (--file--)

# LLega hasta la carpeta donde esta el file y busca en las carpetas de la esturctura pedida 
DATA_PATH = BASE_DIR / "data" / "seguro_auto_actuarial.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42

ARCHIVOS_REQUERIDOS = [
    DATA_PATH,
    MODELS_DIR / "mejor_modelo_regresion.joblib",
    MODELS_DIR / "mejor_modelo_clasificacion.joblib",
    MODELS_DIR / "metadata_modelos.json",
    OUTPUTS_DIR / "comparacion_modelos_regresion.csv",
    OUTPUTS_DIR / "comparacion_modelos_clasificacion.csv",
    OUTPUTS_DIR / "importancias_rf_regresion.csv",
    OUTPUTS_DIR / "importancias_rf_clasificacion.csv",
    BASE_DIR / "utils" / "preprocessing.py",
]

# Si no esta ahí, la mete a la fuerza al inicio
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))



from utils.preprocessing import ActuarialFeatureEngineer # Mete el preporcesamiento.


## Modificación de la pestaña
st.set_page_config(
    page_title="Sistema actuarial de seguros de auto",
    layout="wide"
)

# Verificar los archivos requeridos, si no esta, moestrar error y cuál. luego detener la app.
def validar_archivos_requeridos():
    faltantes = [str(ruta.relative_to(BASE_DIR)) for ruta in ARCHIVOS_REQUERIDOS if not ruta.exists()]
    if faltantes:
        st.error("Faltan archivos requeridos para ejecutar la aplicación.")
        st.write(faltantes)
        st.stop()



# Eliminación de los Outliers, en base a la estrategi usada en el nb.
def aplicar_eliminacion_outliers(df):
    df_filtrado = df.copy()
    columnas_originales = df_filtrado.columns.tolist()

    df_filtrado["log_suma_asegurada"] = np.log1p(
        pd.to_numeric(df_filtrado["suma_asegurada_mxn"], errors="coerce").clip(lower=0)
    )
    df_filtrado["log_costo"] = np.log(
        pd.to_numeric(df_filtrado["costo_esperado_anual_mxn"], errors="coerce").clip(lower=1)
    )
    prima = pd.to_numeric(df_filtrado["prima_mensual_mxn"], errors="coerce").replace(0, np.nan)
    df_filtrado["ratio_costo_prima"] = df_filtrado["costo_esperado_anual_mxn"] / prima

    mascara = (
        (df_filtrado["log_costo"] < df_filtrado["log_costo"].quantile(0.995))
        & (df_filtrado["log_suma_asegurada"] < df_filtrado["log_suma_asegurada"].quantile(0.995))
        & (df_filtrado["ratio_costo_prima"] < df_filtrado["ratio_costo_prima"].quantile(0.995))
    )

    return df_filtrado.loc[mascara, columnas_originales].copy()




@st.cache_data ## Guarda en la memoria 
def cargar_datos(): # Datos
    df = pd.read_csv(DATA_PATH)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return aplicar_eliminacion_outliers(df)

def datos_visualizacion():# Datos para la gráfica interactiva
    df_vis = pd.read_csv(DATA_PATH)
    df_vis = df_vis.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df_vis = aplicar_eliminacion_outliers(df_vis)

    fe = ActuarialFeatureEngineer()
    df_vis = fe.fit(df_vis).transform(df_vis)

    return df_vis

@st.cache_data # Guarda en la memoria
def cargar_tabla(nombre_archivo): # La tabla en outputs
    ruta = OUTPUTS_DIR / nombre_archivo
    return pd.read_csv(ruta)


@st.cache_data # Guarda en la memoria
def cargar_metadata(): # Los datos de los modelos
    with open(MODELS_DIR / "metadata_modelos.json", encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_resource # Guarda en la memoria
def cargar_modelos(): # Cargar los mejores modelos
    regresion = joblib.load(MODELS_DIR / "mejor_modelo_regresion.joblib")
    clasificacion = joblib.load(MODELS_DIR / "mejor_modelo_clasificacion.joblib")
    return regresion, clasificacion


@st.cache_resource # Guarda en la memoria
def cargar_modelo(nombre_archivo): # El modelo que se ocupe
    return joblib.load(MODELS_DIR / nombre_archivo)


def variables_modelo(df):
    df_modelo = df.copy() # Solo las variables usadas, tanto transformaciones como iniciales
    if "costo_esperado_anual_mxn" in df_modelo.columns:
        costo = pd.to_numeric(df_modelo["costo_esperado_anual_mxn"], errors="coerce").clip(lower=1)
        df_modelo["log_costo"] = np.log(costo) # transformacion para elminar outliers
    if {"costo_esperado_anual_mxn", "prima_mensual_mxn"}.issubset(df_modelo.columns):
        prima = pd.to_numeric(df_modelo["prima_mensual_mxn"], errors="coerce").replace(0, np.nan)
        df_modelo["ratio_costo_prima"] = df_modelo["costo_esperado_anual_mxn"] / prima # Tambien para detección de outliers

    excluir = ["poliza_id", "costo_esperado_anual_mxn", "riesgo_alto", "clase_costo"]
    X = df_modelo.drop(columns=[col for col in excluir if col in df_modelo.columns])
    return asegurar_columnas_entrenamiento(X)


def asegurar_columnas_entrenamiento(X):# En caso de que no este, llenarla de nan
    X = X.copy()
    for columna in ["log_costo", "ratio_costo_prima"]:
        if columna not in X.columns:
            X[columna] = np.nan
    return X


def extraer_proba(pipeline, X):# Sacar la probabiliad del pipline del modelo
    probas = pipeline.predict_proba(X)
    clases = list(pipeline.named_steps["model"].classes_)
    return probas[:, clases.index(1)]


def formato_mxn(valor): # pasar a pesos Méxicanos
    return f"${valor:,.0f} MXN"


def limpiar_nombre_feature(nombre): # Para quitar la codificacion
    nombre = nombre.replace("numerico__", "").replace("nominal__", "")
    nombre = nombre.replace("ordinal__", "").replace("binario__", "")
    return nombre


def resumen_preprocesador(pipeline): # Sacar que hizo el pipline
    preprocesador = pipeline.named_steps["preprocesado"]
    filas = []

    for nombre, transformador, columnas in preprocesador.transformers_:
        if nombre == "remainder":
            continue

        pasos = " -> ".join(transformador.named_steps.keys())
        filas.append({
            "bloque": nombre,
            "variables": len(columnas),
            "columnas": ", ".join(columnas),
            "tratamiento": pasos
        })

    return pd.DataFrame(filas)


def modelos_disponibles(metadata, tipo): # 
    return {
        item["modelo"]: item["archivo"]
        for item in metadata["modelos_exportados"]
        if item["tipo"] == tipo
    }


def obtener_importancias(pipeline): # Sacar coeficientes y su importancia.
    modelo = pipeline.named_steps["model"]
    features = pipeline.named_steps["preprocesado"].get_feature_names_out()

    if hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
        columna = "importancia"
    elif hasattr(modelo, "coef_"):
        valores = np.ravel(modelo.coef_)
        columna = "coeficiente"
    else:
        return pd.DataFrame()

    importancias = pd.DataFrame({
        "feature": features,
        columna: valores,
        "magnitud": np.abs(valores)
    })
    importancias["feature"] = importancias["feature"].map(limpiar_nombre_feature)
    return importancias.sort_values("magnitud", ascending=False)


def matriz_confusion_modelo(pipeline, X, y_real, umbral): # Matriz de Confusion
    probas = extraer_proba(pipeline, X)
    pred = (probas >= umbral).astype(int)
    matriz = confusion_matrix(y_real, pred, labels=[0, 1])
    return pd.DataFrame(
        matriz,
        index=["Real bajo", "Real alto"],
        columns=["Pred bajo", "Pred alto"]
    )


def resumen_faltantes(df): # Cosneguir los Nan
    faltantes = pd.DataFrame({
        "variable": df.columns,
        "faltantes": df.isna().sum().values,
        "porcentaje": (100 * df.isna().sum() / len(df)).round(2).values,
        "tipo": df.dtypes.astype(str).values,
        "valores_unicos": df.nunique().values
    })
    return faltantes.sort_values(["faltantes", "variable"], ascending=[False, True])




# Inicio del diseño de la Página

def mostrar_inicio(df, metadata):
    st.title("Sistema actuarial de predicción de riesgo y costo esperado")
    st.write(
        "Aplicación en Streamlit basada en los resultados exportados por "
        "`notebooks/01_eda_modelado.ipynb`. La app carga modelos ya entrenados; "
        "no ajusta modelos predictivos." 


    )
    st.write(
        "Al importar el CSV se aplica la misma eliminación de outliers del notebook: "
        "percentil 99.5 sobre `log_costo`, `log_suma_asegurada` y `ratio_costo_prima`, que es "
        "el cociente entre el costo y su prima. Esto derivado a que al sacar el logaritmo se puede suavizar " \
        "el efecto que los Outliers tienen a la muestra, dándole también sea dicho de paso, un aspecto más simiétrico." \
        "Al sacar aquellos miembros que se encontraban fuera del percentil 99.5, se está sacando a aquellos miembros que inculso con esta " \
        "versión suavizada, siguen muy por afuera de la distribución de la muestra. Se hará el supuesto de que esos registros o bien " \
        "no son correctos, o en su defecto, están gobernados por otro tipo de factores ajenos a la base, es decir," \
        "no son parte de la misma población o forman parte de un sector muy particular de la población que requiere otro " \
        "tratamiento."
    )

    st.write(
        'Sucede que, para cualquier empresa, determinar los costos que va a implicar el desarrollo' \
        'de un proucto es esencial. Muchas veces se tiene una ganancia bruta muy buena,' \
        'pero esta se ve mermada por los costos. En un contexto de seguros varios costos. ' \
        'Pudiendo ser los que se asocian a la operación de la empresa, como el pago de empleados, luz, agua, etc. o bien,' \
        'los derivados de la siniestralidad, que pudieran ser el pago de los distintos beneficios y los medios' \
        'para darlos que se encuentran en su póliza.' )

    st.write(
        'El conocer los costos deriva a una mejor estimación del precio que debe tener una prima, para que ' \
        'se puedan cumplir las diversas promesas que se ven estipuladas en las pólizas.' )
    

    st.write(
        'Saber si una prima es de riesgo alto es esencial para una aseguradora, pues, desde un inicio, ' \
        'puede negarte la cobertura, porque, desde un inicio, se reconoce que la realización del siniestro ya no ' \
        'es algo aleatorio, si no, algo cierto. O en su defecto, cobrar una sobre prima, para ajustar ' \
        'ajustada a la probabiliad de que suceda en particular a dicho cliente.')
    
    st.write(
        ' La base cuenta con 31 columnas, 30 variables, de las cuales 14 se trataron como numéricas y las otras' \
        '16 como categóricas. Inicialmente se contaba con 1500 pólizas. ')
    
    st.subheader(' Objetivo del Proyecto ')
    st.write('El objetivo del proyecto es poder estimar dichos costos, y a su vez, si una poliza puede ser de alto riesgo o no' \
    'en funcion a las variables propuestas por la base, para la parte representaiva de la muestra, o dicho de otras palabras, intentar '
    'estimar para aquellos que no están influidos por otros factores tal vez no explicados por la base o en su defecto, afectados '
    'por condiciones de mero azar.')
    
    st.divider()

    # Acomodar en columnas las caracteristicas del df
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pólizas", f"{len(df):,}")
    col2.metric("Columnas", df.shape[1])
    col3.metric("Mejor regresor", metadata["mejor_modelo_regresion"])
    col4.metric("Mejor clasificador", metadata["mejor_modelo_clasificacion"])

    st.subheader("Archivos usados")
    st.write(
        "- `data/seguro_auto_actuarial.csv`\n"
        "- `models/mejor_modelo_regresion.joblib`\n"
        "- `models/mejor_modelo_clasificacion.joblib`\n"
        "- `models/metadata_modelos.json`\n"
        "- `outputs/comparacion_modelos_*.csv`\n"
        "- `outputs/importancias_rf_*.csv`\n"
        "- `utils/preprocessing.py`"
    )

    st.subheader("Datos sin Outliers")
    st.dataframe(df, use_container_width=True)


def mostrar_exploracion(df):
    st.header("Exploración de datos")


    # Acomodo en columnas 
    col1, col2 = st.columns([1, 1])
    with col1: # LLama a la funcion de faltantes y lo pone.
        st.subheader("Tipos y faltantes")
        st.dataframe(resumen_faltantes(df), use_container_width=True, height=410)

    with col2: # Gráfica interacriva con plotly express. 
        st.subheader("Balance de variables")
        columnas_balance = df.select_dtypes(include=["object", "category"]).columns.tolist()
        columnas_balance += [
            col for col in df.select_dtypes(include=["number"]).columns
            if df[col].nunique() <= 12 and col not in columnas_balance
        ]
        if "poliza_id" in columnas_balance:
            columnas_balance.remove("poliza_id")
        variable_balance = st.selectbox(
            "Variable de balance",
            columnas_balance,
            index=columnas_balance.index("riesgo_alto")
        )
        #  Quita las entradas problematicas. Se hace el conteo del total y de las clases para encontrar el balance.
        balance = df[variable_balance].value_counts(dropna=False).rename_axis(variable_balance).reset_index(name="conteo")
        balance[variable_balance] = balance[variable_balance].replace({0: "Bajo", 1: "Alto"}).astype(str)
        fig = px.bar(
            balance,
            x=variable_balance,
            y="conteo",
            text="conteo",
            color=variable_balance,
            template="plotly_white"
        )
        fig.update_layout(showlegend=False, xaxis_title=variable_balance, yaxis_title="Conteo")
        st.plotly_chart(fig, use_container_width=True)


    ## Columnas numericas.
    st.subheader("Variables creadas en utils/preprocessing.py")
    st.write(
        "- log_ingreso_mensual y log_suma_asegurada\n"
        "- km_totales y log_km_totales\n"
        "- grupo_edad_conductor"
    )
    columnas_numericas = df.select_dtypes(include=["number"]).columns.tolist()
    variable = st.selectbox( # Elegir entre columnas
        "Variable numérica",
        columnas_numericas,
        index=columnas_numericas.index("costo_esperado_anual_mxn")
    )
    # Categoria para elegir y separar en color por la misma
    categorica =['deducible_pct',
                'riesgo_alto',
                'sexo',
                'estado_civil',
                'nivel_estudios',
                'ocupacion',
                'zona_residencia',
                'region',
                'tipo_vehiculo',
                'uso_vehiculo',
                'segmento_marca',
                'metodo_pago',
                'canal_venta',
                'tiene_gps',
                'asistencia_vial',
                'mantenimiento_al_dia',
                'clase_costo',
                'grupo_edad_conductor'] 
    
    # Selecionar la categoria
    vcategorica = st.selectbox('Categoría',
                            categorica)
    

    ## Bara cambiar los rangos de las observaciones para hacer el histograma
    nbins = st.slider('Intervalos de las observaciones',2, 250, 40)
    fig_hist = px.histogram(
        df,
        x=variable,
        color=df[vcategorica],
        marginal="box", # Este parametro pone la boxplots.
        nbins= nbins,
        barmode="overlay",
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Correlaciones numéricas")
    columnas_corr = [col for col in columnas_numericas if col != "riesgo_alto"]
    corr = df[columnas_corr].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        height=850
    
    )
    
    fig_corr.update_traces(xgap=1, ygap=1)
    fig_corr.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_colorbar_title="Corr",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def mostrar_preprocesamiento(df, pipeline_regresion):
    st.header("Preprocesamiento")
    st.write(' Una vez elminados los Outliers de la base, se separa en 3 conjuntos la muestra.' \
    'Uno para entrenar la variables, otro de validación y otro para comparar modelos y seleccionar parametos ' \
    'y finalmente otro para probar si este generaliza. Se estratifico por riesgo para que permanezca estable el balance en estas 3 submuestras'

    )

    st.write(
            'Posteriormente, se construyo un Pipline que: ' )
   
    st.write('Si era númerica, al ya no tener los Outliers de las variables que más relación tenina, se puede' \
            'estandarizar sin mayor problema, y luego, imputar por KNN, con 3 vecinos cercanos.')

    st.write('Si era de otra, solamente imputar por moda. Luego si esta era Ordinal, códificar mediante las listas de orden creadas.' )
    st.write('Si era binaria, codificar el sí y no a 1 y 0.')
    st.write('Despues con esta se almientaron los modelos')

    resumen = resumen_preprocesador(pipeline_regresion)


    matriz = pipeline_regresion.named_steps["preprocesado"].get_feature_names_out()

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables originales", df.shape[1])
    col2.metric("Bloques de transformación", len(resumen))
    col3.metric("Variables después del preprocesamiento", len(matriz))

    st.subheader("Bloques del ColumnTransformer")
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.subheader("Variables creadas en utils/preprocessing.py")
    st.write(
        "- `log_ingreso_mensual` y `log_suma_asegurada`\n"
        "- `km_totales` y `log_km_totales`\n"
        "- `siniestros_en_12m` y `grupo_edad_conductor`"
    )

    st.subheader("Ejemplo de matriz de características")
    vista_features = pd.DataFrame({"feature": matriz[:35]})
    st.dataframe(vista_features, use_container_width=True, hide_index=True)


def mostrar_modelado(df, metadata):
    st.header("Modelado ya exportado")
    st.write(
        "Las tablas provienen de `outputs`. Aquí sólo se leen métricas, modelos e "
        "importancias de los pipelines `.joblib` generados previamente en el notebook."
    )

    reg = cargar_tabla("comparacion_modelos_regresion.csv")
    clf = cargar_tabla("comparacion_modelos_clasificacion.csv")
    modelos_regresion = modelos_disponibles(metadata, "regresion")
    modelos_clasificacion = modelos_disponibles(metadata, "clasificacion")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regresión")
        st.dataframe(reg, use_container_width=True)
        metricas_reg = [col for col in reg.columns if col != "modelo"]
        metrica_reg = st.selectbox("Métrica de regresión", metricas_reg, index=metricas_reg.index("R2"))
        ascendente = metrica_reg in ["MAE", "RMSE"]
        fig = px.bar(
            reg.sort_values(metrica_reg, ascending=ascendente),
            x="modelo",
            y=metrica_reg,
            color="modelo",
            template="plotly_white"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Clasificación")
        st.dataframe(clf, use_container_width=True)
        metricas_clf = [col for col in clf.columns if col != "modelo"]
        metrica_clf = st.selectbox("Métrica de clasificación", metricas_clf, index=metricas_clf.index("F1"))
        ascendente_clf = metrica_clf in ["FP", "FN"]
        fig = px.bar(
            clf.sort_values(metrica_clf, ascending=ascendente_clf),
            x="modelo",
            y=metrica_clf,
            color="modelo",
            template="plotly_white"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Importancias o coeficientes por algoritmo")
    col_tipo, col_modelo, col_top = st.columns([1, 2, 1])
    tipo_importancia = col_tipo.radio("Tipo", ["Regresión", "Clasificación"], horizontal=True)
    modelos_tipo = modelos_regresion if tipo_importancia == "Regresión" else modelos_clasificacion
    
    lista_modelos = list(modelos_tipo.keys())
    key_mejor = "mejor_modelo_regresion" if tipo_importancia == "Regresión" else "mejor_modelo_clasificacion"
    mejor_nombre = metadata[key_mejor]
    idx_default = lista_modelos.index(mejor_nombre) if mejor_nombre in lista_modelos else 0

    modelo_importancia = col_modelo.selectbox("Modelo", lista_modelos, index = idx_default)
    top_n = col_top.slider("Top variables", 5, 25, 12)

    pipeline_importancia = cargar_modelo(modelos_tipo[modelo_importancia])
    importancias = obtener_importancias(pipeline_importancia)
    if importancias.empty:
        st.info("Este algoritmo no expone importancias ni coeficientes interpretables.")
    else:
        columna_valor = "importancia" if "importancia" in importancias.columns else "coeficiente"
        vista = importancias.head(top_n).sort_values("magnitud")
        fig = px.bar(
            vista,
            x=columna_valor,
            y="feature",
            orientation="h",
            template="plotly_white",
            title=f"{modelo_importancia}: principales variables"
        )
        fig.update_layout(yaxis_title="", xaxis_title=columna_valor.capitalize(), height=480)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Diagnóstico visual con modelos guardados")
    st.write("Estas gráficas aplican los modelos ya entrenados sobre la base cargada; no se ajustan modelos nuevos.")
    X = variables_modelo(df)
    col3, col4 = st.columns(2)

    with col3:
        modelo_reg_diagnostico = st.selectbox(
            "Modelo para gráfica de regresión",
            list(modelos_regresion.keys()),
            index=list(modelos_regresion.keys()).index(metadata["mejor_modelo_regresion"])
        )
        pipeline_reg = cargar_modelo(modelos_regresion[modelo_reg_diagnostico])
        y_real = df["costo_esperado_anual_mxn"]
        y_pred = pipeline_reg.predict(X)
        df_reg = pd.DataFrame({"Costo real": y_real, "Costo predicho": y_pred})
        lim_min = min(df_reg["Costo real"].min(), df_reg["Costo predicho"].min())
        lim_max = max(df_reg["Costo real"].max(), df_reg["Costo predicho"].max())
        fig = px.scatter(
            df_reg,
            x="Costo real",
            y="Costo predicho",
            opacity=0.55,
            template="plotly_white",
            title=f"Real vs predicción: {modelo_reg_diagnostico}"
        )
        fig.add_shape(
            type="line",
            x0=lim_min,
            y0=lim_min,
            x1=lim_max,
            y1=lim_max,
            line=dict(color="red", width=2)
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        modelo_clf_diagnostico = st.selectbox(
            "Modelo para matriz de confusión",
            list(modelos_clasificacion.keys()),
            index=list(modelos_clasificacion.keys()).index(metadata["mejor_modelo_clasificacion"])
        )
        pipeline_clf = cargar_modelo(modelos_clasificacion[modelo_clf_diagnostico])
        umbral = float(metadata["umbral_clasificacion"])
        matriz = matriz_confusion_modelo(pipeline_clf, X, df["riesgo_alto"], umbral)
        fig = px.imshow(
            matriz,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title=f"Matriz de confusión: {modelo_clf_diagnostico}"
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Predicción del modelo",
            yaxis_title="Clase real",
            height=460
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Clasificación con umbral operativo: {metadata['umbral_clasificacion']:.2f}. "
        "Este umbral fue seleccionado en validación dentro del notebook."
    )


def calcular_pca_visual(df, pipeline_regresion):
    X = variables_modelo(df)
    feature_engineering = pipeline_regresion.named_steps["feature_engineering"]
    preprocesado = pipeline_regresion.named_steps["preprocesado"]
    matriz = preprocesado.transform(feature_engineering.transform(X))



    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    componentes = pca.fit_transform(matriz)

    df_pca = pd.DataFrame(componentes, columns=["PC1", "PC2"])
    df_pca["riesgo_alto"] = df["riesgo_alto"].map({0: "Bajo", 1: "Alto"}).values
    df_pca["clase_costo"] = df["clase_costo"].values
    return df_pca, pca.explained_variance_ratio_


def mostrar_pca(df, pipeline_regresion):
    st.header("Reducción de dimensionalidad")
    st.write(
        "Para visualizar la cartera, se reutiliza el preprocesador ya ajustado dentro "
        "del mejor modelo de regresión y se calcula una proyección PCA de dos componentes."
    )


    st.write('Principal Component Analysis (PCA) es una técnica de reducción de dimensionalidad no supervisada.' \
    ' Su funcionamiento consiste en proyectar las características originales sobre nuevas direcciones, y calcular su varianza, ' \
    'se quedaran aquellas que maximizan la varianza.' \
    ' La primera componente principal es la combinación lineal de todas las variables originales que captura la máxima varianza posible.' \
    ' La segunda componente es la combinación lineal ortogonal a la primera que captura la máxima varianza restante, y así sucesivamente.' \
    ' La suma de las varianzas de los componentes dira cuanta información se ve preservada en estas ' \
    'compoententes principales.' \
    'En este caso 2 componentes no es suficiente, pues solo preserva el 19% de la informacion.'
    )



    colorear_por = st.radio("Colorear por", ["riesgo_alto", "clase_costo"], horizontal=True)
    df_pca, varianza = calcular_pca_visual(df, pipeline_regresion)

    col1, col2, col3 = st.columns(3)
    col1.metric("Varianza PC1", f"{varianza[0]:.2%}")
    col2.metric("Varianza PC2", f"{varianza[1]:.2%}")
    col3.metric("Varianza total", f"{varianza.sum():.2%}")

    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color=colorear_por,
        opacity=0.70,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def crear_formulario_poliza(df):
    st.subheader("Datos de la póliza")
    datos = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        datos["edad_conductor"] = st.number_input("Edad del conductor", 18, 90, int(df["edad_conductor"].median()))
        datos["sexo"] = st.selectbox("Sexo", sorted(df["sexo"].dropna().unique()))
        datos["estado_civil"] = st.selectbox("Estado civil", sorted(df["estado_civil"].dropna().unique()))
        datos["nivel_estudios"] = st.selectbox(
            "Nivel de estudios",
            ["Secundaria", "Preparatoria", "Licenciatura", "Posgrado"],
            index=2
        )
        datos["ocupacion"] = st.selectbox("Ocupación", sorted(df["ocupacion"].dropna().unique()))
        datos["zona_residencia"] = st.selectbox("Zona de residencia", sorted(df["zona_residencia"].dropna().unique()))
        datos["region"] = st.selectbox("Región", sorted(df["region"].dropna().unique()))
        datos["antiguedad_cliente_anios"] = st.number_input(
            "Antigüedad como cliente",
            0.0,
            40.0,
            float(df["antiguedad_cliente_anios"].median()),
            step=0.1
        )
        datos["ingreso_mensual_mxn"] = st.number_input(
            "Ingreso mensual",
            0.0,
            250000.0,
            float(df["ingreso_mensual_mxn"].median()),
            step=500.0
        )

    with col2:
        datos["score_crediticio"] = st.number_input(
            "Score crediticio",
            300.0,
            850.0,
            float(df["score_crediticio"].median()),
            step=1.0
        )
        datos["prima_mensual_mxn"] = st.number_input(
            "Prima mensual",
            0.0,
            20000.0,
            float(df["prima_mensual_mxn"].median()),
            step=50.0
        )
        datos["suma_asegurada_mxn"] = st.number_input(
            "Suma asegurada",
            0.0,
            3000000.0,
            float(df["suma_asegurada_mxn"].median()),
            step=10000.0
        )
        datos["deducible_pct"] = st.selectbox("Deducible (%)", sorted(df["deducible_pct"].dropna().unique()))
        datos["historial_siniestros_3_anios"] = st.number_input(
            "Siniestros últimos 3 años",
            0,
            10,
            int(df["historial_siniestros_3_anios"].median())
        )
        datos["km_anuales"] = st.number_input(
            "Km anuales",
            0.0,
            100000.0,
            float(df["km_anuales"].median()),
            step=500.0
        )
        datos["edad_vehiculo_anios"] = st.number_input(
            "Edad del vehículo",
            0.0,
            40.0,
            float(df["edad_vehiculo_anios"].median()),
            step=0.1
        )
        datos["tipo_vehiculo"] = st.selectbox("Tipo de vehículo", sorted(df["tipo_vehiculo"].dropna().unique()))
        datos["uso_vehiculo"] = st.selectbox("Uso del vehículo", sorted(df["uso_vehiculo"].dropna().unique()))

    with col3:
        datos["segmento_marca"] = st.selectbox("Segmento de marca", ["Economico", "Medio", "Premium"], index=1)
        datos["metodo_pago"] = st.selectbox("Método de pago", sorted(df["metodo_pago"].dropna().unique()))
        datos["canal_venta"] = st.selectbox("Canal de venta", sorted(df["canal_venta"].dropna().unique()))
        datos["tiene_gps"] = st.selectbox("Tiene GPS", ["No", "Si"])
        datos["asistencia_vial"] = st.selectbox("Asistencia vial", ["No", "Si"])
        datos["mantenimiento_al_dia"] = st.selectbox("Mantenimiento al día", ["No", "Si"])
        datos["dias_hasta_renovacion"] = st.number_input(
            "Días hasta renovación",
            0,
            365,
            int(df["dias_hasta_renovacion"].median())
        )
        datos["puntaje_riesgo_zona"] = st.number_input(
            "Puntaje de riesgo de zona",
            0.0,
            100.0,
            float(df["puntaje_riesgo_zona"].median()),
            step=0.1
        )
        datos["numero_siniestros_12m"] = st.number_input(
            "Siniestros últimos 12 meses",
            0,
            10,
            int(df["numero_siniestros_12m"].median())
        )

    return pd.DataFrame([datos])


def mostrar_simulador(df, pipeline_regresion, pipeline_clasificacion, metadata):
    st.header("Simulador actuarial")
    st.write("La predicción usa directamente los pipelines `.joblib` exportados por el notebook.")

    nueva_poliza = crear_formulario_poliza(df)
    nueva_poliza_modelo = asegurar_columnas_entrenamiento(nueva_poliza)
    pred_costo = float(pipeline_regresion.predict(nueva_poliza_modelo)[0])
    prob_riesgo = float(extraer_proba(pipeline_clasificacion, nueva_poliza_modelo)[0])
    umbral = float(metadata["umbral_clasificacion"])
    clase = "Riesgo alto" if prob_riesgo >= umbral else "Riesgo bajo"

    col1, col2, col3 = st.columns(3)
    col1.metric("Costo esperado anual", formato_mxn(pred_costo))
    col2.metric("Probabilidad de riesgo alto", f"{prob_riesgo:.1%}")
    col3.metric("Clase con umbral", clase)

    st.dataframe(nueva_poliza, use_container_width=True)


def histograma_color(image):
    features = []
    colores = ("Rojo", "Verde", "Azul")

    for i, _ in enumerate(colores):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        features.extend(histogram)

    observation = np.array(features).flatten()
    canal_columna = []
    for color in colores:
        canal_columna.extend([color] * 256)

    return pd.DataFrame({
        "Intensidad": list(range(256)) * 3,
        "Pixeles": observation,
        "Canal": canal_columna
    })


def mostrar_imagenes():
    st.header("Lector de Imágenes")
    st.write("Visualizador de imágenes y sus características")
    archivo = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

    if archivo is None:
        st.info("Sube una imagen para ver histograma, resize, crop, blur, contraste, sharpness y filtro HSV.")
        return

    bytes_imagen = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    image = cv2.imdecode(bytes_imagen, cv2.IMREAD_COLOR)
    if image is None:
        st.error("No se pudo leer la imagen.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    alto_original, ancho_original = image.shape[:2]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image)
    with col2:
        st.subheader("Histograma de color")
        df_hist = histograma_color(image)
        grafico = alt.Chart(df_hist).mark_line(opacity=0.8).encode(
            x=alt.X("Intensidad:Q", title="Intensidad del Píxel (0-255)"),
            y=alt.Y("Pixeles:Q", title="Cantidad de Píxeles", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Canal:N",
                scale=alt.Scale(
                    domain=["Rojo", "Verde", "Azul"],
                    range=["#FF0000", "#00FF00", "#0000FF"]
                )
            )
        ).properties(title="Histograma de Color de la Imagen").interactive()
        st.altair_chart(grafico, use_container_width=True)

    st.divider()
    st.subheader("R E S I Z E")

    st.write('Para trabajar con imagenes, estas deben ser de las mismas dimensiones, y hacerles R E S I Z E, ayuda a cumplir esto, a su vez,' \
    'con el coste de reducir o cambiar la información que esta imagen tiene, ya que al final esta es una matriz.' \
    ' Otra cosa a tener en cuenta es que al hacerle R E S I Z E, ayuda a liberar la memoria de tal vez las miles o cientos de miles de imagenes ' \
    'que estemos usando para poder entrenar nuestro modelo, pues cargar estas puede ser muy pesado.' )
    col1, col2 = st.columns(2)
    with col1:
        ancho = st.slider("Ancho (pixeles)", 10, 800, min(200, max(10, ancho_original)), key="img_resize_ancho")
    with col2:
        alto = st.slider("Alto (pixeles)", 10, 600, min(200, max(10, alto_original)), key="img_resize_alto")

    img_resized = cv2.resize(image, (ancho, alto))
    st.image(img_resized, use_container_width=False)
    st.caption(f"Tamaño: {ancho}x{alto}")

    st.divider()
    st.subheader("C R O P")
    st.write('Puede que nuestras imagenes solo contengan una porcion de interés. C R O P es muy útil para poder especificar' \
    'que parte de la imagen se requiere realmente. Esto se hace facilmente selecciónando las columnas y las filas que queremos de ' \
    'la matriz que usa OpenCV para representarlas.')
    col_x1, col_x2, col_y1, col_y2 = st.columns(4)
    with col_x1:
        x1 = st.slider("Inico de corte a lo ancho", 0, ancho_original, 0, key="img_crop_x1")
    with col_x2:
        x2 = st.slider("Final de corte a lo ancho", 0, ancho_original, ancho_original, key="img_crop_x2")
    with col_y1:
        y1 = st.slider("Inicio de corte a lo alto", 0, alto_original, 0, key="img_crop_y1")
    with col_y2:
        y2 = st.slider("Final del corte a lo alto", 0, alto_original, alto_original, key="img_crop_y2")

    st.caption('NOTA: Si por algún motivo El inicio es más grande que el fin, se invertiran los papeles, y si es igual, no habrá ningún corte.')

    if x2 > x1 and y2 > y1:
        croped = image[y1:y2, x1:x2]
    elif x2 > x1 and y2 < y1:
        croped = image[y2:y1, x1:x2]
    elif x2 < x1 and y2 > y1:
        croped = image[y1:y2, x2:x1]
    else:
        croped = image[y2:y1, x2:x1]

    if x1 != x2 and y1 != y2:
        st.image(croped, use_container_width=True)

    st.divider()
    st.subheader("B L U R")

    st.write('Aunque no muy recomendado, hacerle B L U R  a la imagen sireve para poder reducir el ruido' \
    'que esta contiene. También sirve para poder suavizar bordes y poder quedarse con los bordes importantes, y ' \
    'a veces solo para simplificarla, pues no nos importan los detalles finos. Tambien sirve para mejorar ' \
    'la segmentación que es separar el objeto del fondo. Sin embargo no es recomendado, pues puedes perder información')
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        b1 = st.slider("A lo largo", 1, 100, 1, key="img_blur_b1")
    with col_b2:
        b2 = st.slider("A lo ancho", 1, 100, 1, key="img_blur_b2")
    st.image(cv2.blur(image, (b1, b2)), use_container_width=True)

    st.divider()
    st.subheader("C O N T R A S T")
    
    st.write('Puede servir para aumentar la segmenación entre objetos, alterando la amplitud de la disrbución ' \
    'de gris en regiones amplias. Tiene la desventaja de agregar ruido. ')
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    contrast = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    st.image(contrast, use_container_width=True)

    st.divider()
    st.subheader("S H A R P N E S S")
    st.write('Contrario al B L U R, este realza el pixel en sí aumentanto el detalle de bordes o texturas,' \
    'ayudando a la segmentación y detección de objetos,  pero con el coste es aumentar el ruido, e incluso crear artefactos' \
    'que son anomalias o distorsiones entre la imagen capturada y el mundo fisico.' \
    'A difrencia del Contraste, solo maximiza el contraste entre zonas de alta frecuencia,que es aquellas zonas donde la variación ' \
    'de la velocidad con la que cambian los valores de intensidad es alta,' \
    ' dejando a las que no intactas. ')
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    image_sharp = cv2.filter2D(image, -1, kernel)
    st.image(image_sharp, use_container_width=True)

    st.divider()
    st.subheader("F I L T R O ")
    st.write('Sirve para separar ciertos colores en lo particular, mediante una mascara.')

    st.write('')
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    colores = {
        "Blanco": ([0, 0, 50], [179, 50, 255]),
        "Rojo": ([0, 100, 100], [10, 255, 255]),
        "Verde": ([40, 50, 50], [80, 255, 255]),
        "Azul": ([100, 50, 50], [130, 255, 255]),
        "Amarillo": ([20, 100, 100], [30, 255, 255]),
        "Morado": ([130, 50, 50], [160, 255, 255]),
        "Naranja": ([10, 100, 100], [20, 255, 255]),
        "Cian": ([80, 100, 100], [100, 255, 255])
    }

    color_seleccionado = st.selectbox("Selecciona un color", list(colores.keys()))
    lower, upper = colores[color_seleccionado]
    lower = np.array(lower)
    upper = np.array(upper)

    st.info(
        f"Rangos para {color_seleccionado}: "
        f"Hue {lower[0]}-{upper[0]}, Sat {lower[1]}-{upper[1]}, Val {lower[2]}-{upper[2]}"
    )

    mask = cv2.inRange(img_hsv, lower, upper)
    img_filtrada = cv2.bitwise_and(image, image, mask=mask)
    st.image(img_filtrada, caption=f"Filtro: {color_seleccionado}", use_container_width=True)


def mostrar_conclusiones():
    st.header("Conclusiones")
    st.write(
                '- Se notó un cambio muy significativo en las métricas al retirar los valores que se considerarón atípicos.\n '
                '- El cambio de hacer la partición de los datos desde el inicio también representó una novedad que condujo a tener que pensar de manera distinta.\n' \
                '- Hubo varios problemas de data leakage, debido al tratamiento que se le dio para hacer el EDA, que se pudieron resolver, como el uso de estadístico de toda la base, ' \
                'transformaciones de los datos de toda la base que después se colaron en el entrenamiento, etc.\n'
                '- Como posible mejora podría ser un refinamiento del pipeline, que al ser un nuevo instrumento usado limitó por su parte tal vez parte de la creatividad para realizar ' \
                'el preprocesamiento a los datos. Una mayor soltura con este podría resultar en la implementación de mejores metodologías.\n' 
                '- La falta de costumbre de trabajar en equipo para la realización de códigos, y especialmente los que son largos también fueron una limitante, pues, ' \
                'cada quien tiene una manera de trabajar y razonar distinta, por lo que las implementaciones de ideas, correcciones y mejoras fueron más complicadas de lo necesario por influir ' \
                'en el flujo de trabajo de la aplicación. Desarrollar un plan de trabajo más estructurado, con insumos de entrada y productos, en lo particular, de qué tipo son, resultará luego ' \
                'en menos problemas intermedios.'
    )


def main():
    validar_archivos_requeridos()
    df = cargar_datos()
    df_vis = datos_visualizacion()
    metadata = cargar_metadata()
    pipeline_regresion, pipeline_clasificacion = cargar_modelos()

    st.sidebar.title("Navegación")
    seccion = st.sidebar.radio(
        "Sección",
        [
            "Inicio",
            "Exploración",
            "Preprocesamiento",
            "Modelado",
            "PCA",
            "Simulador",
            "Imágenes",
            "Conclusiones"
        ]
    )

    if seccion == "Inicio":
        mostrar_inicio(df, metadata)
    elif seccion == "Exploración":
        mostrar_exploracion(df_vis)
    elif seccion == "Preprocesamiento":
        mostrar_preprocesamiento(df, pipeline_regresion)
    elif seccion == "Modelado":
        mostrar_modelado(df, metadata)
    elif seccion == "PCA":
        mostrar_pca(df, pipeline_regresion)
    elif seccion == "Simulador":
        mostrar_simulador(df, pipeline_regresion, pipeline_clasificacion, metadata)
    elif seccion == "Imágenes":
        mostrar_imagenes()
    else:
        mostrar_conclusiones()


if __name__ == "__main__":
    main()
