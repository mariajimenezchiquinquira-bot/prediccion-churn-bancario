"""Regresión logística para predecir churn: el modelo más simple, como punto de partida.

Sirve de referencia para comparar: si los demás modelos la superan con claridad,
se justifica usar modelos más complejos.

Flujo típico desde el notebook:

    from src.modelos import regresion_logistica as rl
    busqueda = rl.buscar_hiperparametros(datos)
    rl.resumen_busqueda(busqueda)
    modelo = rl.entrenar_final(datos, rl.mejores_parametros(busqueda))
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.modelos import busqueda as bq
from src.datos.data_prep import RANDOM_STATE, DataBundle

# C es la inversa de la regularización: C chico = modelo más simple, C grande = más libre
PARAM_GRID = {
    "logreg__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    "logreg__class_weight": [None, "balanced"],
}


def construir_pipeline(datos: DataBundle, **params) -> Pipeline:
    """Pipeline: preprocesamiento (con escalado) + regresión logística.

    El escalado hace que la regularización trate igual a todas las variables.
    La regresión logística ya entrega probabilidades calibradas, no necesita calibrarse.
    """
    base = dict(max_iter=2000, random_state=RANDOM_STATE)
    return Pipeline([
        ("pre", datos.preprocessor(scale=True)),
        ("logreg", LogisticRegression(**{**base, **params})),
    ])


def buscar_hiperparametros(datos: DataBundle, param_grid=None, cv: int = 5,
                           n_jobs: int = -1, verbose: int = 1) -> GridSearchCV:
    """Grid search con validación cruzada estratificada SOLO sobre el set de entrenamiento."""
    return bq.ejecutar_busqueda(GridSearchCV, construir_pipeline(datos), datos, cv, n_jobs,
                                verbose, param_grid=param_grid or PARAM_GRID)


def mejores_parametros(busqueda: GridSearchCV) -> dict:
    """Mejores hiperparámetros sin el prefijo 'logreg__'."""
    return bq.mejores_parametros(busqueda, "logreg__")


def resumen_busqueda(busqueda: GridSearchCV, top: int = 10) -> pd.DataFrame:
    """Las mejores combinaciones según PR-AUC promedio en validación cruzada."""
    r = pd.DataFrame(busqueda.cv_results_)
    cols = {
        "param_logreg__C": "C",
        "param_logreg__class_weight": "class_weight",
        "mean_test_pr_auc": "pr_auc",
        "std_test_pr_auc": "pr_auc_std",
        "mean_test_roc_auc": "roc_auc",
        "mean_test_recall": "recall",
        "mean_test_f1": "f1",
        "mean_fit_time": "seg_ajuste",
    }
    out = r[list(cols)].rename(columns=cols)
    out["class_weight"] = out["class_weight"].fillna("ninguno")
    return out.sort_values("pr_auc", ascending=False).head(top).round(4).reset_index(drop=True)


def entrenar_final(datos: DataBundle, params: dict) -> Pipeline:
    """Reentrena con los mejores parámetros sobre todo el set de entrenamiento."""
    return bq.entrenar_final(construir_pipeline(datos, **params), datos)


def coeficientes(modelo: Pipeline, top: int = 10) -> pd.DataFrame:
    """Variables con mayor peso (en valor absoluto). Positivo = sube la probabilidad de fuga."""
    nombres = modelo["pre"].get_feature_names_out()
    coef = pd.DataFrame({"variable": nombres, "coeficiente": modelo["logreg"].coef_[0]})
    coef["abs"] = coef["coeficiente"].abs()
    return (coef.sort_values("abs", ascending=False).head(top)
            .drop(columns="abs").round(3).reset_index(drop=True))
