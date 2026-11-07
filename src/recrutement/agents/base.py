"""
Socle commun des agents.

Un agent =  un périmètre de variables  +  un apprentissage sur Train  +  un avis structuré.
L'avis (classe Avis) contient toujours : une note /100, une confiance [0, 1] et des arguments.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config as C


@dataclass
class Avis:
    agent: str
    note: float | None              # note sur 100 (None pour un agent de contrôle)
    confiance: float                # 0 = hésitant, 1 = très sûr
    points_forts: list = field(default_factory=list)
    points_faibles: list = field(default_factory=list)
    texte: str = ""
    details: dict = field(default_factory=dict)


class Agent(ABC):
    nom: str = "Agent"
    critere: str | None = None      # critère MCDM alimenté par l'agent

    def __init__(self, variables: list):
        self.variables = list(variables)
        self.entraine = False

    def fit(self, train: pd.DataFrame) -> "Agent":
        self.entraine = True
        return self

    @abstractmethod
    def evaluer(self, candidat: dict) -> Avis:
        """Avis détaillé sur un candidat (dictionnaire variable -> valeur)."""

    def noter(self, df: pd.DataFrame) -> np.ndarray:
        """Notes seules, pour un lot de candidats (version rapide, sans argumentaire)."""
        return np.array([self.evaluer(r).note for r in df.to_dict("records")])

    @staticmethod
    def confiance(p: float) -> float:
        """Confiance = netteté de l'avis : 0 si p = 0.5 (hésitation), 1 si p = 0 ou 1."""
        return round(float(abs(2 * p - 1)), 3)

    @staticmethod
    def libelle(variable: str, valeur) -> str:
        return f"{C.LABELS.get(variable, variable)} = {valeur:g}"
