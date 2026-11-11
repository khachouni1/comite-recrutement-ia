"""Tests du découpage : aucune offre partagée entre Train, Validation et Test (pas de fuite)."""
import pytest

from recrutement import data


@pytest.fixture(scope="module")
def jeux():
    tr, te = data.charger_brut()
    return data.decouper(data.preparer(tr), data.preparer(te))


def test_aucune_offre_partagee(jeux):
    train, val, test = jeux
    assert not set(train["offre_id"]) & set(val["offre_id"])
    assert not set(train["offre_id"]) & set(test["offre_id"])
    assert not set(val["offre_id"]) & set(test["offre_id"])


def test_tailles_et_etiquettes(jeux):
    train, val, test = jeux
    assert len(train) + len(val) == 6240 and len(test) == 1759
    for part in jeux:
        assert set(part["y"]) == {0, 1, 2}


def test_textes_anonymises(jeux):
    assert not jeux[0]["cv"].str.contains(r"\S+@\S+\.\S+", regex=True).any()
