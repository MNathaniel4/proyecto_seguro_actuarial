# Sistema actuarial de prediccion de riesgo y costo esperado en seguros de automovil

Aplicacion en Streamlit para explorar una cartera sintetica de seguros de automovil, evaluar modelos de Machine Learning y simular predicciones actuariales. El proyecto estima:

- `costo_esperado_anual_mxn`: costo anual esperado de siniestros.
- `riesgo_alto`: clasificacion binaria de polizas de alto riesgo.

La base `seguro_auto_actuarial.csv` es sintetica, academica y no contiene datos personales reales.

## Contenido del proyecto

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- seguro_auto_actuarial.csv
|-- notebooks/
|   `-- 01_eda_modelado.ipynb
|-- models/
|   |-- clasificacion_arbol_balanceado.joblib
|   |-- clasificacion_dummy_prior.joblib
|   |-- clasificacion_gradient_boosting.joblib
|   |-- clasificacion_logistica_balanceada.joblib
|   |-- clasificacion_random_forest_balanceado.joblib
|   |-- mejor_modelo_clasificacion.joblib
|   |-- mejor_modelo_regresion.joblib
|   |-- metadata_modelos.json
|   |-- regresion_arbol.joblib
|   |-- regresion_ridge.joblib
|-- outputs/
|   |-- comparacion_modelos_regresion.csv
|   |-- comparacion_modelos_clasificacion.csv
|   |-- importancias_rf_regresion.csv
|   `-- importancias_rf_clasificacion.csv
`-- utils/
|   |-- preprocessing.py
|--assets
|   | --Carro.jpg
|   | --OIP.jpg  
```

## Instalacion


Si se hace por anaconda: 

Abra Anaconda Promt y vaya a la carpeta raíz con `cd`. 
Se recomienda la creación de un nuevo ambiente.

```bash
conda create --name ambiente  
```
Se recommienda que sea `python==3.11.14`.


Activar ambiente:
```bash
conda activate ambiente  
```


Si por algún motivo no estuviera `pip` instalado ya:
```bash
conda install pip
```


Instalar dependencias:
```bash
pip install -r requirements.txt
```





De otra forma: 

Se recomienda usar un entorno virtual.

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt


```

## Ejecucion de la aplicacion

Desde la carpeta raiz del proyecto:

```bash
streamlit run app.py
```

La app valida que existan los archivos requeridos en `data/`, `models/`, `outputs/` y `utils/`. Si falta algun archivo, Streamlit mostrara una lista de elementos faltantes y detendra la ejecucion.__NOTA: ESTOS ARCHIVOS SE GENERAN AL EJECUTAR EL NOTEBOOK__ . En este caso ya vienen cargados al repositorios los resultados.

## Modulos principales

- **Inicio**: contexto del problema actuarial, objetivos y resumen de los modelos.
- **Exploracion de datos**: vista de datos, tipos de variables, faltantes, distribuciones, correlaciones y balance de clases.
- **Preprocesamiento**: transformaciones numericas y categoricas, imputacion, escalamiento, codificacion y variables nuevas.
- **Modelado**: comparacion de modelos de regresion y clasificacion, metricas, matriz de confusion e importancias.
- **Reduccion de dimensionalidad**: PCA de dos componentes para visualizar polizas por `riesgo_alto` o `clase_costo`.
- **Simulador actuarial**: formulario para estimar costo esperado y probabilidad/clase de riesgo alto de una poliza nueva.
- **Imagenes**: carga de imagen y transformaciones basicas como resize, crop, blur, contraste, sharpening, histograma y filtro de color.
- **Conclusiones**: hallazgos, limitaciones y mejoras futuras.

## Metodologia

El flujo sigue las indicaciones del proyecto:

1. Comprension del problema y variables objetivo.
2. Exploracion de datos, faltantes, outliers y balance de `riesgo_alto`.
3. Preprocesamiento con `Pipeline` y `ColumnTransformer`.
4. Ingenieria de variables con `ActuarialFeatureEngineer`.
5. Entrenamiento y comparacion de modelos lineales, regularizados, arboles, Random Forest y Gradient Boosting.
6. Evaluacion con metricas de regresion y clasificacion.
7. Visualizacion PCA.
8. Integracion de resultados y simulador en Streamlit.

## Modelos incluidos

### Regresion

- Dummy mediana
- Regresion lineal
- Ridge
- Lasso
- Arbol de decision
- Random Forest
- Gradient Boosting

El mejor modelo exportado para regresion es `models/mejor_modelo_regresion.joblib`.

### Clasificacion

- Dummy prior
- Regresion logistica balanceada
- Arbol balanceado
- Random Forest balanceado
- Gradient Boosting

El mejor modelo exportado para clasificacion es `models/mejor_modelo_clasificacion.joblib`. La app usa el umbral guardado en `models/metadata_modelos.json`.

## Resultados guardados

Los archivos de `outputs/` contienen:

- comparacion de modelos de regresion: MAE, RMSE y R2.
- comparacion de modelos de clasificacion: accuracy, precision, recall, F1, ROC AUC, PR AUC y matriz de confusion.
- importancias de variables para Random Forest de regresion y clasificacion.

## Notebook

El notebook `notebooks/01_eda_modelado.ipynb` contiene el analisis exploratorio, preprocesamiento, entrenamiento, evaluacion y exportacion de modelos. Para abrirlo:

```bash
jupyter notebook notebooks/01_eda_modelado.ipynb
```

## Notas de reproducibilidad

- Ejecutar siempre desde la raiz del proyecto para que las rutas relativas funcionen correctamente.
- La aplicacion carga modelos ya entrenados desde `models/`.
- Si se reentrena desde el notebook, conservar los nombres esperados por `app.py` o actualizar `models/metadata_modelos.json`.
- La base principal debe estar en `data/seguro_auto_actuarial.csv`.

## Entregables relacionados

El proyecto se complementa con un reporte breve y un video demostrativo mostrando la app funcionando, las decisiones metodologicas, los modelos, el simulador y el modulo de imagenes.
