
"""Sprint 1, regresión: entrena el modelo de popularidad (Ridge) y escribe models/modelo.pkl.
 
Regresión Ridge sobre las variables disponibles antes del lanzamiento, más duración al
cuadrado, entrenada con las canciones de 1995 a 2019 y evaluada con las de 2020 a 2025.
El alpha se elige con validación cruzada temporal dentro del periodo de entrenamiento.
 
    python -m src.entrenar_regresion
 
Para usar su propio modelo, cambie `construir_pipeline`, `NOMBRE` y, si lo decide, `CORTE_PROMOCION`.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
=======
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
>>>>>>> 227667f56d1b89487e6cc557f948c49c17dbb577
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.artefacto import guardar
from src.variables import (
    COLUMNAS_ENTRADA,
    cargar_canciones,
    derivadas,
    particion_temporal,
    preprocesamiento,
    resumen_periodo,
)

<<<<<<< HEAD
NOMBRE = "ridge + duración²"
=======
NOMBRE = "lineal + duración²"
>>>>>>> 227667f56d1b89487e6cc557f948c49c17dbb577
CORTE_PROMOCION = 65  # popularidad esperada a partir de la cual se recomienda promocionar
VARIABLES_EXCLUIDAS = {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "es_hit": "se calcula a partir del objetivo (popularidad >= 70)",
}


def construir_pipeline():
    """Debe recibir las COLUMNAS_ENTRADA crudas: el preprocesamiento va dentro del pipeline."""
<<<<<<< HEAD
    return make_pipeline(
        FunctionTransformer(derivadas),
        preprocesamiento(),
        RidgeCV(
            alphas=np.logspace(-3, 3, 50),
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_absolute_error",
        ),
    )
=======
    return make_pipeline(FunctionTransformer(derivadas), preprocesamiento(), LinearRegression())
>>>>>>> 227667f56d1b89487e6cc557f948c49c17dbb577


def entrenar() -> dict:
    datos = cargar_canciones()
    entrenamiento, prueba = particion_temporal(datos)
    X, y = datos[COLUMNAS_ENTRADA], datos["popularidad"]

    pipeline = construir_pipeline().fit(X[entrenamiento], y[entrenamiento])
    pred = pipeline.predict(X[prueba])
    residuo = y[prueba] - pred

<<<<<<< HEAD
    # Coeficientes de Ridge: ninguno llega a cero; se reportan los 5 de mayor peso.
    ridge = pipeline[-1]
    nombres = pipeline[-2].get_feature_names_out()
    coefs = pd.Series(ridge.coef_, index=nombres)
    mayores = coefs.abs().sort_values(ascending=False).head(5).index

    # Línea base: media por género calculada solo con el periodo de entrenamiento.
    # Si aparece un género nuevo en prueba, se usa la media global de entrenamiento.
    media_genero = y[entrenamiento].groupby(datos.loc[entrenamiento, "genero"]).mean()
    base = datos.loc[prueba, "genero"].map(media_genero).fillna(y[entrenamiento].mean())
=======
    # Línea base: media por género calculada solo con el periodo de entrenamiento.
    media_genero = y[entrenamiento].groupby(datos.loc[entrenamiento, "genero"]).mean()
    base = datos.loc[prueba, "genero"].map(media_genero)
>>>>>>> 227667f56d1b89487e6cc557f948c49c17dbb577

    por_genero = (
        pd.DataFrame({"genero": datos.loc[prueba, "genero"], "abs": residuo.abs(), "res": residuo})
        .groupby("genero")
        .agg(mae=("abs", "mean"), sesgo=("res", "mean"))
        .round(2)
    )

    ficha = {
        "nombre": NOMBRE,
        "tarea": "regresion",
        "objetivo": "popularidad (0-100) que alcanzará la canción",
        "momento_prediccion": "antes del lanzamiento: la disquera decide cuánto invertir en promoción",
        "variables_entrada": list(COLUMNAS_ENTRADA),
        "variables_excluidas": VARIABLES_EXCLUIDAS,
        "entrenamiento": resumen_periodo(datos, entrenamiento),
        "prueba": {**resumen_periodo(datos, prueba), "tipo": "temporal"},
        "linea_base": {
            "descripcion": "media por género (entrenamiento)",
            "mae": round(float(mean_absolute_error(y[prueba], base)), 2),
        },
        "desempeno": {
            "mae": round(float(mean_absolute_error(y[prueba], pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y[prueba], pred))), 2),
            "r2": round(float(r2_score(y[prueba], pred)), 3),
            "sesgo": round(float(residuo.mean()), 2),
        },
<<<<<<< HEAD
        "hiperparametros": {"alpha": round(float(ridge.alpha_), 5)},
        "coeficientes_principales": {str(n): round(float(coefs[n]), 3) for n in mayores},
=======
>>>>>>> 227667f56d1b89487e6cc557f948c49c17dbb577
        "por_genero": por_genero.to_dict(orient="index"),
        "decision": {
            "corte_promocion": CORTE_PROMOCION,
            "regla": "promocionar si la popularidad esperada supera el corte",
            "nota": "el corte lo fija quien responde por el presupuesto, no el modelo",
        },
        "limites": "describe asociaciones en datos sintéticos; no es evidencia causal",
    }
    return guardar(pipeline, ficha)


if __name__ == "__main__":
<<<<<<< HEAD
    entrenar()
=======
    entrenar()
>>>>>>> 227667f56d1b89487e6cc557f948c49c17dbb577
