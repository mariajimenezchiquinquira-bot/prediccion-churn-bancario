"""Modelos de ensamble para predecir churn.

- Bagging        -> árboles entrenados con muestras aleatorias de clientes.
- Random Forest  -> bagging de árboles, eligiendo variables al azar en cada división.
- XGBoost        -> boosting de árboles.
- LightGBM       -> boosting de árboles (variante más rápida).
- Stacking       -> combina SVM + ensambles con un meta-modelo logístico.

Flujo típico desde el notebook:

    from src.modelos import ensemble_models as ens
    busq = ens.buscar_hiperparametros("Random Forest", datos)
    ens.resumen_busqueda(busq)
    modelo = ens.entrenar_final("Random Forest", datos, ens.mejores_parametros(busq))
"""
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.modelos import busqueda as bq
from src.datos.data_prep import RANDOM_STATE, DataBundle
from src.modelos.svm_model import crear_svc

NOMBRES = ["Bagging", "Random Forest", "XGBoost", "LightGBM"]
# Modelos que entran al Stacking junto con el SVM (Bagging queda fuera: se parece mucho al Random Forest)
BASE_STACKING = ["Random Forest", "XGBoost", "LightGBM"]


# --------------------------------------------------------------------------
# Construcción de modelos
# --------------------------------------------------------------------------
def _clasificador(nombre: str, params: dict | None = None):
    """Clasificador base (sin preprocesamiento) con valores por defecto + params.

    n_jobs=1 en cada modelo: el paralelismo se hace a nivel de búsqueda.
    """
    params = dict(params or {})
    if nombre == "Bagging":
        # Bagging "puro": árboles que ven TODAS las variables en cada división;
        # la única fuente de variación es el bootstrap de las filas.
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=1,
        )
        return bag.set_params(**params)   # set_params admite estimator__max_depth, etc.
    if nombre == "Random Forest":
        base = dict(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)
        return RandomForestClassifier(**{**base, **params})
    if nombre == "XGBoost":
        base = dict(n_estimators=300, eval_metric="logloss", tree_method="hist",
                    random_state=RANDOM_STATE, n_jobs=1)
        return XGBClassifier(**{**base, **params})
    if nombre == "LightGBM":
        # subsample solo actúa en LightGBM si subsample_freq > 0
        base = dict(n_estimators=300, subsample_freq=1, random_state=RANDOM_STATE,
                    n_jobs=1, verbose=-1)
        return LGBMClassifier(**{**base, **params})
    if nombre == "SVM":
        return crear_svc(**params)
    raise ValueError(f"Modelo desconocido: {nombre}")


def construir_pipeline(nombre: str, datos: DataBundle, calibrar: bool = False,
                       **params) -> Pipeline:
    """Pipeline: preprocesamiento (sin escalado, los árboles no lo necesitan) + modelo.

    Con `calibrar=True` el modelo se envuelve en CalibratedClassifierCV, lo que
    corrige las probabilidades (los pesos de clase las inflan) y las deja listas
    para leerse como probabilidad real de fuga.
    """
    clf = _clasificador(nombre, params)
    if calibrar:
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=5, ensemble=False)
    return Pipeline([("pre", datos.preprocessor(scale=False)), ("clf", clf)])


# --------------------------------------------------------------------------
# Búsqueda de hiperparámetros
# --------------------------------------------------------------------------
def espacio_busqueda(nombre: str, datos: DataBundle) -> dict:
    """Distribuciones de hiperparámetros a explorar (búsqueda aleatoria)."""
    y = datos.y_train
    peso = float((y == 0).sum() / (y == 1).sum())   # ~5.2 con 16% de churn
    if nombre == "Bagging":
        return {
            "clf__n_estimators": randint(50, 200),
            "clf__max_samples": uniform(0.5, 0.5),          # fracción de filas por árbol: 0.5 a 1.0
            "clf__estimator__max_depth": [8, 12, 16],
            "clf__estimator__min_samples_leaf": randint(1, 6),
            "clf__estimator__class_weight": [None, "balanced"],
        }
    if nombre == "Random Forest":
        return {
            "clf__n_estimators": randint(200, 600),
            "clf__max_depth": [None, 8, 12, 20],
            "clf__min_samples_leaf": randint(1, 8),
            "clf__max_features": ["sqrt", 0.3, 0.5],
            "clf__class_weight": [None, "balanced_subsample"],
        }
    if nombre == "XGBoost":
        return {
            "clf__n_estimators": randint(150, 500),
            "clf__max_depth": randint(3, 8),
            "clf__learning_rate": loguniform(0.02, 0.2),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__min_child_weight": randint(1, 8),
            "clf__reg_lambda": loguniform(0.5, 10),
            "clf__scale_pos_weight": [1.0, peso],
        }
    if nombre == "LightGBM":
        return {
            "clf__n_estimators": randint(150, 500),
            "clf__num_leaves": randint(15, 64),
            "clf__learning_rate": loguniform(0.02, 0.2),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__min_child_samples": randint(10, 60),
            "clf__reg_lambda": loguniform(0.1, 10),
            "clf__class_weight": [None, "balanced"],
        }
    raise ValueError(f"Modelo desconocido: {nombre}")


def buscar_hiperparametros(nombre: str, datos: DataBundle, n_iter: int = 20, cv: int = 5,
                           n_jobs: int = -1, verbose: int = 1) -> RandomizedSearchCV:
    """Búsqueda aleatoria con validación cruzada estratificada SOLO sobre el train."""
    return bq.ejecutar_busqueda(RandomizedSearchCV, construir_pipeline(nombre, datos), datos,
                                cv, n_jobs, verbose, param_distributions=espacio_busqueda(nombre, datos),
                                n_iter=n_iter, random_state=RANDOM_STATE)


def mejores_parametros(busqueda: RandomizedSearchCV) -> dict:
    """Mejores hiperparámetros sin el prefijo 'clf__'."""
    return bq.mejores_parametros(busqueda, "clf__")


def resumen_busqueda(busqueda: RandomizedSearchCV, top: int = 8) -> pd.DataFrame:
    """Mejores combinaciones según PR-AUC promedio en validación cruzada."""
    r = pd.DataFrame(busqueda.cv_results_)
    params = pd.DataFrame(list(r["params"])).rename(columns=lambda c: c.replace("clf__", ""))
    metr = pd.DataFrame({
        "pr_auc": r["mean_test_pr_auc"],
        "pr_auc_std": r["std_test_pr_auc"],
        "roc_auc": r["mean_test_roc_auc"],
        "recall": r["mean_test_recall"],
        "f1": r["mean_test_f1"],
        "seg_ajuste": r["mean_fit_time"],
    })
    # Presentación: None se muestra como "None" (pandas lo convierte en NaN) y
    # los enteros guardados como float (12.0) vuelven a verse como enteros.
    def _mostrar(v):
        if pd.isna(v):
            return "None"
        if isinstance(v, float):
            return int(v) if v.is_integer() else round(v, 4)
        return v

    params = params.astype(object).apply(lambda col: col.map(_mostrar))
    out = pd.concat([metr.round(4), params], axis=1).sort_values("pr_auc", ascending=False)
    return out.head(top).reset_index(drop=True)


def entrenar_final(nombre: str, datos: DataBundle, params: dict) -> Pipeline:
    """Reentrena con los mejores parámetros sobre todo el train, con probabilidades calibradas."""
    return bq.entrenar_final(construir_pipeline(nombre, datos, calibrar=True, **params), datos)


# --------------------------------------------------------------------------
# Stacking
# --------------------------------------------------------------------------
def entrenar_stacking(datos: DataBundle, params_modelos: dict) -> Pipeline:
    """Stacking: los modelos base generan predicciones fuera de muestra (CV) y un
    meta-modelo (regresión logística) aprende a combinarlas.

    `params_modelos` = {"SVM": {...}, "Random Forest": {...}, "XGBoost": {...}, "LightGBM": {...}}
    con los mejores hiperparámetros de cada uno (sin prefijo).
    Se usa escalado porque el SVM lo necesita; no perjudica a los árboles.
    """
    base = [(n, _clasificador(n, p)) for n, p in params_modelos.items()]
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="auto",
        n_jobs=1,
    )
    modelo = Pipeline([("pre", datos.preprocessor(scale=True)), ("clf", stack)])
    return modelo.fit(datos.X_train, datos.y_train)
