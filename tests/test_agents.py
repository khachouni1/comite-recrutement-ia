"""Tests du comité : équité par construction, cohérence AHP, décisions."""
import numpy as np
import pytest

from recrutement import config as C
from recrutement import data
from recrutement.agents import Coordinateur, agent_tech


@pytest.fixture(scope="module")
def jeux():
    return data.split_three_way(data.load_raw())


@pytest.fixture(scope="module")
def comite(jeux):
    if not (C.RESULTS_DIR / "03_modele_retenu.json").exists():
        pytest.skip("Exécutez d'abord scripts/03_supervise.py")
    return Coordinateur().fit(jeux[0])


def test_aucun_agent_evaluateur_n_utilise_genre_ou_age(comite):
    for agent in comite.evaluateurs.values():
        assert not set(agent.variables) & set(C.PROTECTED), agent.nom


def test_contrefactuel_genre_age_sans_effet(comite, jeux):
    res = comite.ethique.test_contrefactuel(jeux[2])
    assert res["contrefactuel_ok"].all()


def test_avis_complet(jeux):
    agent = agent_tech().fit(jeux[0])
    avis = agent.evaluer(jeux[2].iloc[0].to_dict())
    assert 0 <= avis.note <= 100 and 0 <= avis.confiance <= 1 and avis.texte


def test_matrice_ahp_incoherente_refusee():
    with pytest.raises(ValueError):
        Coordinateur(matrice_ahp=[[1, 9, 9, 1 / 9], [1 / 9, 1, 9, 9], [1 / 9, 1 / 9, 1, 9], [9, 1 / 9, 1 / 9, 1]])


def test_decisions_et_short_list(comite, jeux):
    test = jeux[2].head(60)
    dec = comite.decider(comite.deliberer(test, expliquer=False))
    assert set(dec["decision"]) <= {"Recommandé", "À examiner", "Écarté (véto)", "Non retenu"}
    assert dec["selectionne"].sum() <= comite.nb_places(len(test))
    assert not dec.loc[dec["veto"] != "", "selectionne"].any()   # un candidat sous véto n'est jamais retenu
    assert np.isfinite(dec["TOPSIS_score"]).all()
