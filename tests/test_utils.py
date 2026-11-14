# --------------------------------------------------------------------------
# Pruebas de la caché y del guardado de modelos
# --------------------------------------------------------------------------
from src import utils


def test_cacheado_calcula_una_sola_vez(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "CACHE_DIR", tmp_path)
    llamadas = []

    def calcular():
        llamadas.append(1)
        return {"valor": 42}

    assert utils.cacheado("prueba", calcular) == {"valor": 42}
    assert utils.cacheado("prueba", calcular) == {"valor": 42}
    assert len(llamadas) == 1


def test_cacheado_con_forzar_vuelve_a_calcular(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "CACHE_DIR", tmp_path)
    utils.cacheado("prueba", lambda: 1)
    assert utils.cacheado("prueba", lambda: 2, forzar=True) == 2
    assert utils.cacheado("prueba", lambda: 3) == 2


def test_guardar_y_cargar_modelos(tmp_path, monkeypatch, sintetico):
    monkeypatch.setattr(utils, "MODELS_DIR", tmp_path)
    modelo, X_te, _, _, _ = sintetico
    utils.guardar_modelos({"Mi modelo": modelo}, {"Mi modelo": {"dato": 1}})
    modelos, resultados = utils.cargar_modelos()
    assert resultados == {"Mi modelo": {"dato": 1}}
    assert (modelos["Mi modelo"].predict_proba(X_te) == modelo.predict_proba(X_te)).all()
