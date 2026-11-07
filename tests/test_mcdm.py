"""Tests des méthodes MCDM sur des cas dont on connaît la réponse."""
import numpy as np
import pandas as pd
import pytest

from recrutement.mcdm import AHP, sensitivity, topsis, wsm

CRIT = ["a", "b", "c"]


def test_ahp_matrice_parfaitement_coherente():
    # poids attendus 0.6 / 0.3 / 0.1  ->  a_ij = w_i / w_j  : matrice parfaitement cohérente
    w = np.array([0.6, 0.3, 0.1])
    ahp = AHP(CRIT, w[:, None] / w[None, :])
    assert np.allclose(ahp.weights_array, w)
    assert ahp.lambda_max == pytest.approx(3.0)
    assert ahp.cr == pytest.approx(0.0, abs=1e-9)
    assert ahp.is_consistent


def test_ahp_detecte_incoherence():
    # a > b (9), b > c (9) mais c > a (9) : jugements contradictoires
    m = [[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]]
    assert not AHP(CRIT, m).is_consistent


def test_ahp_refuse_matrice_non_reciproque():
    with pytest.raises(ValueError):
        AHP(CRIT, [[1, 2, 3], [2, 1, 1], [1 / 3, 1, 1]])


def test_ahp_from_judgments_equivaut_a_la_matrice():
    m = [[1, 2, 4], [1 / 2, 1, 2], [1 / 4, 1 / 2, 1]]
    a1 = AHP(CRIT, m)
    a2 = AHP.from_judgments(CRIT, {("a", "b"): 2, ("a", "c"): 4, ("b", "c"): 2})
    assert np.allclose(a1.weights_array, a2.weights_array)


SCORES = pd.DataFrame({"a": [90, 50, 10], "b": [80, 50, 20], "c": [70, 50, 30]}, index=["X", "Y", "Z"])
POIDS = {"a": 0.5, "b": 0.3, "c": 0.2}


def test_topsis_alternative_dominante_premiere():
    res = topsis(SCORES, POIDS)
    assert list(res.sort_values("TOPSIS_rang").index) == ["X", "Y", "Z"]
    assert res.loc["X", "TOPSIS_score"] == pytest.approx(1.0)   # X est la solution idéale
    assert res.loc["Z", "TOPSIS_score"] == pytest.approx(0.0)   # Z est la solution anti-idéale


def test_wsm_bornes_et_ordre():
    res = wsm(SCORES, POIDS)
    assert res.loc["X", "WSM_score"] == pytest.approx(1.0)
    assert res.loc["Z", "WSM_score"] == pytest.approx(0.0)
    assert res["WSM_rang"].tolist() == [1, 2, 3]


def test_sensibilite_sans_variation_ne_change_rien():
    s = sensitivity(SCORES, POIDS, top_k=1, deltas=(0.0,))
    assert (s["recouvrement_topK"] == 1).all()
    assert np.allclose(s["spearman"], 1)
