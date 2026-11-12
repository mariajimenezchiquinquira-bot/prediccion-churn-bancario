"""Pruebas de que cada tipo de modelo se entrena y entrega probabilidades válidas."""
import numpy as np
import pytest

from src.modelos import ensemble_models as ens
from src import modelo_keras as mk
from src.modelos import regresion_logistica as rl
from src.modelos import svm_model as svm


def _revisar_probabilidades(modelo, datos):
    p = modelo.predict_proba(datos.X_test)
    assert p.shape == (len(datos.X_test), 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)


def test_regresion_logistica(datos_pequenos):
    modelo = rl.entrenar_final(datos_pequenos, {"C": 1, "class_weight": "balanced"})
    assert modelo.steps[0][0] == "pre"          # el preprocesador va dentro del Pipeline
    _revisar_probabilidades(modelo, datos_pequenos)


def test_svm_calibrado(datos_pequenos):
    modelo = svm.entrenar_final(datos_pequenos, {"C": 1, "gamma": "scale", "class_weight": None})
    _revisar_probabilidades(modelo, datos_pequenos)


@pytest.mark.parametrize("nombre", ens.NOMBRES)
def test_ensambles(nombre, datos_pequenos):
    params = {"n_estimators": 20}
    modelo = ens.entrenar_final(nombre, datos_pequenos, params)
    _revisar_probabilidades(modelo, datos_pequenos)


def test_red_neuronal(datos_pequenos):
    modelo = mk.entrenar_final(datos_pequenos, {"epocas": 2, "capas": (8,)})
    _revisar_probabilidades(modelo, datos_pequenos)
    assert len(modelo.named_steps["red"].historia_.history["loss"]) <= 2


def test_la_red_rechaza_un_optimizador_desconocido(datos_pequenos):
    with pytest.raises(ValueError):
        mk.entrenar_final(datos_pequenos, {"optimizador": "otro", "epocas": 1})
