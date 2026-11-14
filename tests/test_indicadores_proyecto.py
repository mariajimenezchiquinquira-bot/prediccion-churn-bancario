# --------------------------------------------------------------------------
# Pruebas sobre los resultados guardados: ¿el proyecto cumple lo que se
# pidió?
#
# Necesitan models/resultados.joblib, que se crea al correr main.ipynb
# completo.
# --------------------------------------------------------------------------
import pytest

from src import evaluation as ev
from src import utils

ARCHIVO = utils.MODELS_DIR / "resultados.joblib"
pytestmark = pytest.mark.skipif(not ARCHIVO.exists(), reason="Falta correr main.ipynb")

TECNICAS = ["SVM", "Regresion logistica", "Bagging", "Random Forest", "XGBoost",
            "LightGBM", "Stacking", "Red neuronal"]


@pytest.fixture(scope="module")
def resultados():
    import joblib
    return joblib.load(ARCHIVO)


def test_estan_todos_los_modelos_pedidos(resultados):
    for nombre in TECNICAS:
        assert nombre in resultados


def test_todos_evaluados_sobre_la_misma_prueba(resultados):
    tamanos = {len(r["y_true"]) for r in resultados.values()}
    assert tamanos == {2026}


def test_todos_superan_el_azar(resultados):
    base = next(iter(resultados.values()))["y_true"].mean()   # PR-AUC del azar = tasa de churn
    for nombre, r in resultados.items():
        assert r["metricas"]["pr_auc"] > 2 * base, nombre


def test_el_modelo_recomendado_cumple_las_metas_del_banco(resultados):
    ind = ev.indicadores_negocio(resultados["LightGBM"])
    for indicador, valor in ind.items():
        assert ev.cumple_meta(indicador, valor), f"{indicador}: {valor}"


def test_el_modelo_recomendado_es_de_los_mejores(resultados):
    tabla = ev.tabla_comparativa(resultados)
    assert "LightGBM" in list(tabla.index[:3])
