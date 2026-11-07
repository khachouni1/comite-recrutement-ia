"""
AHP — Analytic Hierarchy Process (Saaty).

Entrée : matrice de comparaisons par paires A (n × n), réciproque : a_ji = 1 / a_ij.
  1. Poids w = vecteur propre principal de A (normalisé pour que Σ w = 1) : A·w = λmax·w
  2. Indice de cohérence     CI = (λmax − n) / (n − 1)
  3. Ratio de cohérence      CR = CI / RI(n)   (RI = indice aléatoire de Saaty)
  Les jugements sont acceptés si CR < 0.10.
"""
from itertools import combinations

import numpy as np

RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


class AHP:
    def __init__(self, criteria: list, matrix):
        self.criteria = list(criteria)
        self.matrix = np.asarray(matrix, dtype=float)
        n = len(self.criteria)
        if self.matrix.shape != (n, n):
            raise ValueError(f"La matrice doit être de taille {n}×{n} (une ligne par critère, dans l'ordre).")
        if np.any(self.matrix <= 0) or not np.allclose(self.matrix * self.matrix.T, 1.0, atol=1e-6):
            raise ValueError("Matrice AHP invalide : valeurs > 0 et réciprocité a_ji = 1/a_ij requises.")
        self._compute()

    @classmethod
    def from_judgments(cls, criteria: list, judgments: dict) -> "AHP":
        """Construit la matrice à partir de jugements nommés {(critère_i, critère_j): a_ij}."""
        idx = {c: k for k, c in enumerate(criteria)}
        m = np.eye(len(criteria))
        for (ci, cj), v in judgments.items():
            m[idx[ci], idx[cj]], m[idx[cj], idx[ci]] = v, 1.0 / v
        return cls(criteria, m)

    def _compute(self) -> None:
        eigvals, eigvecs = np.linalg.eig(self.matrix)
        k = int(np.argmax(eigvals.real))
        w = np.abs(eigvecs[:, k].real)
        self.weights_array = w / w.sum()
        self.lambda_max = float(eigvals[k].real)
        n = len(self.criteria)
        self.ci = (self.lambda_max - n) / (n - 1) if n > 2 else 0.0
        self.cr = self.ci / RANDOM_INDEX[n] if RANDOM_INDEX.get(n, 0) > 0 else 0.0
        self.is_consistent = self.cr < 0.10

    @property
    def weights(self) -> dict:
        return dict(zip(self.criteria, self.weights_array))

    def pairs(self):
        """Paires (i, j) avec i < j : les jugements à saisir (n(n-1)/2)."""
        return list(combinations(self.criteria, 2))

    def summary(self) -> dict:
        return {"poids": {c: round(float(w), 4) for c, w in self.weights.items()},
                "lambda_max": round(self.lambda_max, 4), "CI": round(self.ci, 4),
                "CR": round(self.cr, 4), "coherent": bool(self.is_consistent)}
