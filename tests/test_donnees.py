"""Tests du découpage Train / Val / Test : tailles, stratification et absence de fuite."""
import pytest

from recrutement import config as C
from recrutement import data


@pytest.fixture(scope="module")
def decoupage():
    return data.split_three_way(data.load_raw())


def test_proportions_70_15_15(decoupage):
    train, val, test = decoupage
    n = len(train) + len(val) + len(test)
    assert n == 1500
    assert len(test) / n == pytest.approx(0.15, abs=0.01)
    assert len(val) / n == pytest.approx(0.15, abs=0.01)


def test_aucun_candidat_partage(decoupage):
    train, val, test = decoupage
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)


def test_stratification(decoupage):
    taux = [part[C.TARGET].mean() for part in decoupage]
    assert max(taux) - min(taux) < 0.01


def test_attributs_proteges_exclus_des_modeles():
    assert not set(C.PROTECTED) & set(C.MODEL_FEATURES)
