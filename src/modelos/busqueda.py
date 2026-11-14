# --------------------------------------------------------------------------
# Pasos comunes de la búsqueda de hiperparámetros, usados por svm_model y
# ensemble_models.
#
# Los dejo en un solo lugar para no repetir el mismo código en los dos
# módulos.
# --------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold

from src.datos.data_prep import RANDOM_STATE, DataBundle

# Se optimiza PR-AUC (más informativa que accuracy con 16% de churn); las demás
# métricas se registran para el informe.
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "f1": "f1",
}


# Ejecuto la búsqueda (GridSearchCV o RandomizedSearchCV) con validación
# cruzada estratificada, solo sobre el entrenamiento.
#
# El preprocesador va dentro del pipeline, por eso se ajusta en cada fold
# sin fuga de información.
#
# `espacio` recibe lo propio de cada búsqueda (param_grid, o
# param_distributions con n_iter y random_state).
def ejecutar_busqueda(clase, pipeline, datos: DataBundle, cv: int = 5, n_jobs: int = -1,
                      verbose: int = 1, **espacio):
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    busqueda = clase(pipeline, scoring=SCORING, refit="pr_auc", cv=skf,
                     n_jobs=n_jobs, verbose=verbose, **espacio)
    return busqueda.fit(datos.X_train, datos.y_train)


# Convierto los tipos de numpy a tipos normales de Python (para imprimir y
# reutilizar)
def _limpio(v):
    return v.item() if hasattr(v, "item") else v


# Devuelvo los mejores hiperparámetros sin el prefijo del paso del pipeline
# ('svc__' o 'clf__')
def mejores_parametros(busqueda, prefijo: str) -> dict:
    return {k.replace(prefijo, ""): _limpio(v) for k, v in busqueda.best_params_.items()}


# Reentreno el pipeline (con probabilidades calibradas) con todo el
# entrenamiento
def entrenar_final(modelo, datos: DataBundle):
    return modelo.fit(datos.X_train, datos.y_train)
