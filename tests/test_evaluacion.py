"""Pruebas de las métricas, los umbrales y las tablas de evaluación."""
import numpy as np
import pytest

from src import evaluation as ev


def test_umbral_f1_esta_entre_0_y_1(sintetico):
    modelo, _, _, X_tr, y_tr = sintetico
    umbral = ev.elegir_umbral(y_tr, ev.puntaje(modelo, X_tr), "f1")
    assert 0 < umbral < 1


def test_umbral_por_recall_alcanza_el_recall_pedido(sintetico):
    modelo, _, _, X_tr, y_tr = sintetico
    s = ev.puntaje(modelo, X_tr)
    umbral = ev.elegir_umbral(y_tr, s, "recall", recall_objetivo=0.80)
    recall = ((s >= umbral) & (y_tr == 1)).sum() / y_tr.sum()
    assert recall >= 0.80


def test_umbral_imposible_da_error(sintetico):
    modelo, _, _, X_tr, y_tr = sintetico
    with pytest.raises(ValueError):
        ev.elegir_umbral(y_tr, ev.puntaje(modelo, X_tr), "recall", recall_objetivo=1.5)


def test_criterio_desconocido_da_error(sintetico):
    modelo, _, _, X_tr, y_tr = sintetico
    with pytest.raises(ValueError):
        ev.elegir_umbral(y_tr, ev.puntaje(modelo, X_tr), "otro")


def test_umbral_por_cv_es_reproducible(sintetico):
    modelo, _, _, X_tr, y_tr = sintetico
    a = ev.umbral_por_cv(modelo, X_tr, y_tr, n_jobs=1)
    b = ev.umbral_por_cv(modelo, X_tr, y_tr, n_jobs=1)
    assert a == b


def test_evaluar_modelo_devuelve_metricas_validas(sintetico):
    modelo, X_te, y_te, _, _ = sintetico
    r = ev.evaluar_modelo("Prueba", modelo, X_te, y_te, umbral=0.4)
    assert set(r["metricas"]) == {"accuracy", "balanced_acc", "precision", "recall",
                                  "f1", "roc_auc", "pr_auc"}
    assert all(0 <= v <= 1 for v in r["metricas"].values())
    np.testing.assert_array_equal(r["y_pred"], (r["y_score"] >= 0.4).astype(int))


def test_tabla_comparativa_ordena_por_pr_auc(sintetico):
    modelo, X_te, y_te, _, _ = sintetico
    resultados = {"A": ev.evaluar_modelo("A", modelo, X_te, y_te, 0.5),
                  "B": ev.evaluar_modelo("B", modelo, X_te[:300], y_te[:300], 0.3)}
    tabla = ev.tabla_comparativa(resultados)
    assert list(tabla["pr_auc"]) == sorted(tabla["pr_auc"], reverse=True)


def test_tabla_lift_captura_todas_las_fugas_al_final(sintetico):
    modelo, X_te, y_te, _, _ = sintetico
    tabla = ev.tabla_lift(ev.evaluar_modelo("A", modelo, X_te, y_te, 0.5))
    assert len(tabla) == 10
    assert tabla["clientes"].sum() == len(y_te)
    assert tabla["fugas"].sum() == y_te.sum()
    assert tabla["captura_acum_%"].iloc[-1] == 100
    assert tabla["lift"].iloc[0] > 1


def _resultado_perfecto(n=1000, tasa=0.16):
    y = np.zeros(n, dtype=int)
    y[: int(n * tasa)] = 1
    s = y.astype(float)
    pred = y.copy()
    metricas = {"recall": 1.0, "precision": 1.0, "pr_auc": 1.0}
    return {"y_true": y, "y_score": s, "y_pred": pred, "metricas": metricas, "umbral": 0.5}


def test_indicadores_de_un_modelo_perfecto():
    ind = ev.indicadores_negocio(_resultado_perfecto())
    assert ind["recall"] == 1 and ind["precision"] == 1
    assert ind["contactados_%"] == pytest.approx(16.0)
    assert ind["captura_20%"] == pytest.approx(100.0)
    assert all(ev.cumple_meta(k, v) for k, v in ind.items())


def test_cumple_meta_respeta_el_sentido_de_cada_meta():
    assert ev.cumple_meta("recall", 0.90)
    assert not ev.cumple_meta("recall", 0.50)
    assert ev.cumple_meta("contactados_%", 15)       # "a lo sumo"
    assert not ev.cumple_meta("contactados_%", 35)


def test_tabla_indicadores_trae_la_meta_y_cuenta_las_cumplidas():
    tabla = ev.tabla_indicadores({"Perfecto": _resultado_perfecto()})
    assert list(tabla.index) == ["Meta", "Perfecto"]
    assert tabla.loc["Perfecto", "metas_cumplidas"] == "5 de 5"
