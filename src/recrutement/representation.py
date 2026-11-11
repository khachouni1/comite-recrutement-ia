"""
Représentations vectorielles du texte, de la plus simple à la plus avancée :

1. TF-IDF (baseline) : chaque mot pondéré par sa fréquence dans le document et sa rareté dans le corpus.
       tfidf(t, d) = tf(t, d) · log(N / df(t))
2. LSA (Latent Semantic Analysis) : décomposition en valeurs singulières tronquée de la matrice TF-IDF.
   C'est l'équivalent de l'ACP pour le texte (Cours 2, réduction de dimension) : on garde les
   k « thèmes » qui expliquent le plus de variance.
3. Embeddings de phrases (Hugging Face, sentence-transformers) : réseau de neurones pré-entraîné
   qui place des textes de même sens à proximité, même sans mots communs.

La similarité CV / offre est mesurée par le cosinus :  cos(u, v) = u·v / (‖u‖ ‖v‖).
Tous les apprentissages (vocabulaire TF-IDF, axes LSA) sont faits sur le TRAIN uniquement.
"""
import hashlib
import os
import pickle
import warnings

# Le protocole de téléchargement « Xet » de Hugging Face s'est bloqué lors des essais : on utilise le HTTP classique.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config as C

CACHE = C.DATA_PROCESSED / "cache_embeddings.pkl"


def cosinus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosinus ligne à ligne entre deux matrices de même taille."""
    return np.sum(normalize(a) * normalize(b), axis=1)


class Representations:
    def __init__(self, utiliser_embeddings: bool = True):
        self.utiliser_embeddings = utiliser_embeddings
        self._encodeur = None
        self._cache = {}

    # ------------------------------------------------------------ TF-IDF + LSA (appris sur Train)
    def fit(self, textes_train) -> "Representations":
        self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.9,
                                     ngram_range=(1, 2), max_features=30000)
        X = self.tfidf.fit_transform(textes_train)
        self.lsa = TruncatedSVD(n_components=C.LSA_COMPONENTS, random_state=C.RANDOM_STATE).fit(X)
        return self

    def tfidf_vec(self, textes):
        return self.tfidf.transform(textes)

    def lsa_vec(self, textes) -> np.ndarray:
        return self.lsa.transform(self.tfidf.transform(textes))

    # ------------------------------------------------------------ embeddings Hugging Face
    def _charger_encodeur(self):
        if self._encodeur is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encodeur = SentenceTransformer(C.EMBEDDING_MODEL, device="cpu")
                if CACHE.exists():
                    self._cache = pickle.loads(CACHE.read_bytes())
            except Exception as e:  # pas de connexion / bibliothèque absente -> repli sur LSA
                warnings.warn(f"Embeddings indisponibles ({e}) : la LSA est utilisée à la place.")
                self.utiliser_embeddings = False
        return self._encodeur

    @staticmethod
    def _morceaux(texte: str, taille: int = 180) -> list:
        """Le modèle lit au plus ~256 jetons : on découpe le texte et on moyenne les morceaux."""
        mots = texte.split()
        return [" ".join(mots[i:i + taille]) for i in range(0, max(len(mots), 1), taille)][:12]

    def embedding_vec(self, textes) -> np.ndarray:
        textes = list(textes)
        if not self.utiliser_embeddings or self._charger_encodeur() is None:
            return self.lsa_vec(textes)
        cles = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in textes]
        # chaque texte DISTINCT n'est encodé qu'une fois (un même CV apparaît dans plusieurs paires)
        manquants = {k: t for k, t in zip(cles, textes) if k not in self._cache}
        if manquants:
            morceaux, proprietaire = [], []
            for k, t in manquants.items():
                m = self._morceaux(t)
                morceaux += m
                proprietaire += [k] * len(m)
            vecs = self._encodeur.encode(morceaux, batch_size=64, normalize_embeddings=True)  # encodage par lots
            proprietaire = np.array(proprietaire)
            for k in manquants:
                self._cache[k] = vecs[proprietaire == k].mean(axis=0)
        return np.vstack([self._cache[k] for k in cles])

    def sauver_cache(self) -> None:
        if self._cache:
            CACHE.write_bytes(pickle.dumps(self._cache))

    def __getstate__(self):  # l'encodeur (réseau de neurones) n'est pas sauvegardé, il est rechargé
        return {k: v for k, v in self.__dict__.items() if k not in ("_encodeur", "_cache")} | \
            {"_encodeur": None, "_cache": {}}

    # ------------------------------------------------------------ similarités d'une liste de paires
    def similarites(self, cvs, offres) -> dict:
        cvs, offres = list(cvs), list(offres)
        t_cv, t_of = self.tfidf_vec(cvs), self.tfidf_vec(offres)
        sim_tfidf = np.asarray(t_cv.multiply(t_of).sum(axis=1)).ravel()  # vecteurs TF-IDF déjà normés
        sim_lsa = cosinus(self.lsa.transform(t_cv), self.lsa.transform(t_of))
        sim_emb = cosinus(self.embedding_vec(cvs), self.embedding_vec(offres))
        return {"sim_tfidf": sim_tfidf, "sim_lsa": sim_lsa, "sim_embedding": sim_emb}
