"""
Socle commun des agents.

Un agent = un périmètre de variables + un apprentissage sur Train + un avis structuré (classe Avis) :
une note sur 100, une confiance entre 0 et 1, des points forts, des points faibles et un texte.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Avis:
    agent: str
    critere: str | None
    note: float | None
    confiance: float
    points_forts: list = field(default_factory=list)
    points_faibles: list = field(default_factory=list)
    texte: str = ""
    details: dict = field(default_factory=dict)


class Agent(ABC):
    nom = "Agent"
    critere: str | None = None
    variables: list = []

    @abstractmethod
    def fit(self, feats: pd.DataFrame, y: pd.Series) -> "Agent":
        """Apprentissage sur le Train."""

    @abstractmethod
    def noter(self, feats: pd.DataFrame) -> np.ndarray:
        """Notes /100 d'un lot de paires (rapide, sans argumentaire)."""

    @abstractmethod
    def evaluer(self, feats: pd.DataFrame, details: dict) -> Avis:
        """Avis argumenté sur UNE paire (feats : 1 ligne, details : éléments extraits du texte)."""

    @staticmethod
    def confiance(p: float) -> float:
        """0 = hésitation (p = 0.5), 1 = avis tranché (p = 0 ou 1)."""
        return round(float(abs(2 * p - 1)), 3)
