# --------------------------------------------------------------------------
# Pruebas de la carga, la limpieza y la partición de los datos
# --------------------------------------------------------------------------
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.datos import data_prep as dp


def test_limpieza_deja_el_formato_esperado(df):
    assert df.shape == (10127, 18)
    assert df.index.name == dp.ID_COL
    assert df.index.is_unique
    assert dp.TARGET in df.columns
    assert dp.TARGET_ORIGINAL not in df.columns
    assert set(df[dp.TARGET].unique()) == {0, 1}


def test_limpieza_quita_las_variables_redundantes(df):
    for col in dp.REDUNDANT_COLS:
        assert col not in df.columns


def test_porcentaje_de_churn_es_el_esperado(df):
    assert df[dp.TARGET].mean() == pytest.approx(0.161, abs=0.002)


def test_particion_es_80_20(datos, df):
    total = len(df)
    assert len(datos.X_train) + len(datos.X_test) == total
    assert len(datos.X_test) / total == pytest.approx(0.2, abs=0.001)
    assert len(datos.X_train) == len(datos.y_train)
    assert len(datos.X_test) == len(datos.y_test)


def test_particion_no_mezcla_entrenamiento_y_prueba(datos):
    assert set(datos.X_train.index).isdisjoint(datos.X_test.index)


def test_particion_conserva_el_porcentaje_de_churn(datos):
    assert datos.y_train.mean() == pytest.approx(datos.y_test.mean(), abs=0.002)


def test_el_target_no_esta_en_las_variables(datos):
    assert dp.TARGET not in datos.X_train.columns
    assert dp.TARGET not in datos.X_test.columns


def test_particion_es_reproducible(df, datos):
    otra = dp.prepare_data(df)
    assert list(otra.X_test.index) == list(datos.X_test.index)


def test_el_preprocesador_no_se_ajusta_con_todos_los_datos(datos):
    with pytest.raises(NotFittedError):
        check_is_fitted(datos.preprocessor())


def test_el_escalado_se_aprende_solo_con_el_entrenamiento(datos):
    pre = datos.preprocessor().fit(datos.X_train)
    escalador = pre.named_transformers_["num"]
    np.testing.assert_allclose(escalador.mean_, datos.X_train[datos.numericas].mean().to_numpy())


def test_falla_si_el_archivo_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError):
        dp.load_data(tmp_path / "no_existe.xlsx")


def test_verificacion_de_rangos_no_encuentra_negativos(df):
    rango = dp.verificar_valores_numericos(df)
    assert (rango["minimo"] >= 0).all()
    assert {"credit_limit", "total_trans_amt"} <= set(rango.index)


def test_verificacion_falla_si_hay_valores_negativos(df):
    malo = df.copy()
    malo.iloc[0, malo.columns.get_loc("credit_limit")] = -1
    with pytest.raises(ValueError):
        dp.verificar_valores_numericos(malo)
