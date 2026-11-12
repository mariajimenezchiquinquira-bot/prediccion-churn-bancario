"""Piezas comunes de la búsqueda de hiperparámetros, usadas por svm_model y ensemble_models.

Se dejan en un solo lugar para no repetir el mismo código en los dos módulos.
"""
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


def ejecutar_busqueda(clase, pipeline, datos: DataBundle, cv: int = 5, n_jobs: int = -1,
                      verbose: int = 1, **espacio):
    """Búsqueda (GridSearchCV o RandomizedSearchCV) con validación cruzada estratificada
    SOLO sobre el set de entrenamiento.

    El preprocesador va dentro del pipeline, así que se ajusta en cada fold con datos de
    entrenamiento únicamente (sin fuga de información). `espacio` recibe lo propio de cada
    búsqueda (param_grid, o param_distributions con n_iter y random_state).
    """
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    busqueda = clase(pipeline, scoring=SCORING, refit="pr_auc", cv=skf,
                     n_jobs=n_jobs, verbose=verbose, **espacio)
    return busqueda.fit(datos.X_train, datos.y_train)


def _limpio(v):
    """Convierte tipos numpy a tipos nativos de Python (para imprimir y reutilizar)."""
    return v.item() if hasattr(v, "item") else v


def mejores_parametros(busqueda, prefijo: str) -> dict:
    """Mejores hiperparámetros sin el prefijo del paso del pipeline ('svc__' o 'clf__')."""
    return {k.replace(prefijo, ""): _limpio(v) for k, v in busqueda.best_params_.items()}


def entrenar_final(modelo, datos: DataBundle):
    """Reentrena el pipeline (con probabilidades calibradas) sobre todo el set de entrenamiento."""
    return modelo.fit(datos.X_train, datos.y_train)
