
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ActuarialFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        # Esta parte se entrena únicamente con lo recibido en fit para evitar data leakage
        prima = pd.to_numeric(X["prima_mensual_mxn"], errors="coerce")
        self.prima_mediana_ = prima.median()
        return self

    def transform(self, X):
        X = X.copy()

        ingreso = pd.to_numeric(X["ingreso_mensual_mxn"], errors='coerce').clip(lower=0)
        suma = pd.to_numeric(X["suma_asegurada_mxn"], errors='coerce').clip(lower=0)
        edad = pd.to_numeric(X["edad_conductor"], errors='coerce')
        edad_carro = pd.to_numeric(X['edad_vehiculo_anios'], errors='coerce')
        km_anuales = pd.to_numeric(X['km_anuales'], errors='coerce')
        prima = pd.to_numeric(X["prima_mensual_mxn"], errors = 'coerce')
        siniestros_12m = pd.to_numeric(X["numero_siniestros_12m"], errors = 'coerce')

        X["log_ingreso_mensual"] = np.log1p(ingreso)
        X["log_suma_asegurada"] = np.log1p(suma)

        X['km_totales'] = edad_carro * km_anuales
        X['log_km_totales'] = np.log1p(X["km_totales"])

        X['siniestros_en_12m'] = np.where(siniestros_12m > 0, "Si", "No")
        X['prima_mayor_a_mediana'] = np.where(prima > self.prima_mediana_, "Si", "No")

        X['grupo_edad_conductor'] = pd.cut(
            edad,
            bins=[17, 25, 35, 50, 65, np.inf],
            labels=['18-25', '26-35', '36-50', '51-65', '66+'],
            include_lowest=True
        ).astype(object)


        return X
