"""Tests du comité (nécessitent le pipeline exécuté : models/comite.joblib)."""
import joblib
import numpy as np
import pytest

from recrutement import config as C
from recrutement.agents import Coordinateur

CV = ("Senior Data Analyst with 7 years of experience in SQL, Python, Tableau and Excel. Built dashboards and "
      "reporting for finance stakeholders. Strong communication, attention to detail. Master of Science.")
OFFRE = ("We are hiring a Data Analyst. Requirements: 3+ years of experience with SQL and Python, Tableau or "
         "Power BI, strong communication skills. Bachelor's degree required.")
CV_HORS_SUJET = ("Certified electrician with 15 years of experience in electrical systems, OSHA, blueprints and "
                 "maintenance of HVAC equipment. High school diploma.")


@pytest.fixture(scope="module")
def comite():
    path = C.MODELS_DIR / "comite.joblib"
    if not path.exists():
        pytest.skip("Exécutez d'abord scripts/run_all.py")
    return joblib.load(path)


def test_rapport_complet(comite):
    res = comite.evaluer(CV, OFFRE)
    assert set(res["avis"]) == set(C.CRITERIA)
    assert all(0 <= a.note <= 100 and 0 <= a.confiance <= 1 for a in res["avis"].values())
    assert 0 <= res["score_global"] <= 100
    assert res["decision"] in {"Recommandé", "À examiner", "Non retenu", "Écarté (véto)"}


def test_candidat_pertinent_mieux_note_que_hors_sujet(comite):
    bon, hors = comite.evaluer(CV, OFFRE, test_ethique=False), comite.evaluer(CV_HORS_SUJET, OFFRE, test_ethique=False)
    assert bon["score_global"] > hors["score_global"]
    assert bon["avis"]["score_technique"].note > hors["avis"]["score_technique"].note


def test_contrefactuel_ethique(comite):
    res = comite.evaluer(CV, OFFRE)
    assert res["ethique"].details["contrefactuel"]["ok"]


def test_classement_plusieurs_cv(comite):
    out = comite.classer({"pertinent": CV, "hors_sujet": CV_HORS_SUJET}, OFFRE)
    assert list(out.index) == ["pertinent", "hors_sujet"]
    assert np.isfinite(out["score_global"]).all()


def test_ahp_incoherent_refuse(comite):
    # cycle de préférences contradictoires : chaque critère domine le suivant, le dernier domine le premier
    n = len(C.CRITERIA)
    m = np.ones((n, n))
    for i in range(n):
        j = (i + 1) % n
        m[i, j], m[j, i] = 9, 1 / 9
    with pytest.raises(ValueError):
        Coordinateur(comite.reps, matrice_ahp=m)
