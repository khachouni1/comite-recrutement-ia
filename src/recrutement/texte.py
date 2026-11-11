"""
Traitement du texte : lecture des fichiers, nettoyage, anonymisation et extraction de critères.

Toutes les fonctions sont déterministes et explicables (règles + lexiques) : chaque critère
peut être justifié au candidat (« compétence trouvée », « 5 ans exigés, 3 ans détectés »…).
"""
import io
import re
from datetime import date

from .lexiques import DEGREES, HARD_SKILLS, SOFT_SKILLS

CURRENT_YEAR = date.today().year


# =========================================================================== lecture & nettoyage
def lire_fichier(nom: str, contenu: bytes) -> str:
    """Lit un CV ou une offre déposé sur le site : .pdf, .docx ou texte brut."""
    nom = nom.lower()
    if nom.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(contenu)).pages)
    if nom.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(contenu)).paragraphs)
    return contenu.decode("utf-8", errors="ignore")


def nettoyer(texte: str) -> str:
    """Normalise les espaces et sépare les mots collés (« SummaryCertified » -> « Summary Certified »)."""
    texte = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(texte))
    return re.sub(r"\s+", " ", texte).strip()


# =========================================================================== anonymisation (éthique)
# Informations « fortes » : la phrase entière qui les contient est supprimée (si elle est courte).
_SENSIBLES = {
    "date de naissance": r"\b(date of birth|dob|born on|born in|birth ?date)\b",
    "âge": r"\bage\s*[:\-]?\s*\d{2}\b|\b\d{2}\s*(years? old|yo)\b",
    "situation familiale": r"\b(married|divorced|widowed|marital status)\b",
    "parentalité": r"\b(mother|father|parent) of\b|\b(children|kids|pregnan\w*)\b",
    "genre": r"\b(gender|sex)\s*:|\b(woman|man|male|female)\b",
    "nationalité / religion": r"\b(nationality|citizenship|religion|citizen of)\b",
}
# Informations « ponctuelles » : seul le mot ou le motif est retiré.
_JETONS = {
    "e-mail": r"\S+@\S+\.\S+",
    "téléphone": r"\+?\d[\d\s().-]{7,}\d",
    "lien / photo": r"https?://\S+|www\.\S+",
    "civilité": r"\b(mr|mrs|ms|miss|mister|madam)\b\.?",
    "pronoms genrés": r"\b(he|she|him|her|his|hers|himself|herself)\b",
}
_ANONYMISATION = list(_SENSIBLES.items()) + list(_JETONS.items())  # (nom, motif), pour le rapport éthique
_MOTS_VIDES = {"is", "a", "an", "the", "and", "of", "i", "am", "was", "with", "to", "in", "at", "on", "none"}


def anonymiser(texte: str, longueur_max_phrase: int = 40) -> str:
    """
    Retire les informations personnelles et protégées AVANT toute évaluation :
    le comité ne peut pas utiliser ce qu'il ne voit pas (équité par construction).

    1. Toute phrase courte (≤ 40 mots) contenant une information « forte » (âge, date de naissance,
       situation familiale, parentalité, genre, nationalité, religion) est supprimée en entier ;
       dans une phrase plus longue, seul le motif est retiré (pour ne pas effacer le contenu professionnel).
    2. Les informations ponctuelles (e-mail, téléphone, lien, civilité, pronoms genrés) sont retirées.
    3. Les fragments devenus vides (« is a , . ») sont supprimés : ils ne doivent laisser aucune trace.
    """
    phrases = re.split(r"(?<=[.!?;|])\s+", texte)
    gardees = []
    for p in phrases:
        for motif in _SENSIBLES.values():
            if re.search(motif, p, re.IGNORECASE):
                p = "" if len(p.split()) <= longueur_max_phrase else re.sub(motif, " ", p, flags=re.IGNORECASE)
        for motif in _JETONS.values():
            p = re.sub(motif, " ", p, flags=re.IGNORECASE)
        p = re.sub(r"\s*([,.;:])(\s*[,.;:])+", r"\1", p)  # ponctuation orpheline « , , . »
        mots = re.findall(r"[A-Za-z]+", p)
        if mots and not all(m.lower() in _MOTS_VIDES for m in mots):
            gardees.append(p.strip(" ,;:"))
    return re.sub(r"\s+", " ", " ".join(gardees)).strip()


# =========================================================================== extraction de critères
def _compile(lexique: dict) -> dict:
    return {nom: re.compile(r"(?<![\w+#])(" + "|".join(variantes) + r")(?![\w+#])", re.IGNORECASE)
            for nom, variantes in lexique.items()}


_HARD = _compile(HARD_SKILLS)
_SOFT = _compile(SOFT_SKILLS)
_DEGREES = [(niveau, re.compile(r"\b(" + "|".join(v) + r")\b", re.IGNORECASE)) for niveau, v in DEGREES]


def competences(texte: str, type_: str = "hard") -> set:
    lex = _HARD if type_ == "hard" else _SOFT
    return {nom for nom, motif in lex.items() if motif.search(texte)}


def niveau_diplome(texte: str, exigence: bool = False) -> int:
    """
    CV : plus haut diplôme mentionné.  Offre (exigence=True) : plus BAS diplôme mentionné,
    car « bachelor required, master preferred » exige un bachelor.
    """
    niveaux = [n for n, motif in _DEGREES if motif.search(texte)]
    if not niveaux:
        return 0
    return min(niveaux) if exigence else max(niveaux)


_NOMBRES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20}
_ANNEES_EXPLICITES = re.compile(
    r"(\d{1,2}|" + "|".join(_NOMBRES) + r")\s*\(?\d{0,2}\)?\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)
_PERIODES = re.compile(r"((?:19|20)\d{2})\s*(?:-|–|to|until)\s*((?:19|20)\d{2}|present|current|now|today)",
                       re.IGNORECASE)


def _valeur(token: str) -> int:
    return int(token) if token.isdigit() else _NOMBRES[token.lower()]


def annees_experience_cv(texte: str) -> float:
    """
    Deux indices, on garde le plus grand :
      - mention explicite (« 13 years of experience », « over twelve (12) years ») ;
      - étendue des périodes d'emploi datées (« 2012 - Present ») : de la première à la dernière année.
    """
    explicites = [_valeur(m.group(1)) for m in _ANNEES_EXPLICITES.finditer(texte)]
    explicites = [v for v in explicites if 0 < v <= 45]
    debuts, fins = [], []
    for d, f in _PERIODES.findall(texte):
        d = int(d)
        f = CURRENT_YEAR if not f[:2].isdigit() else int(f)
        if 1970 <= d <= f <= CURRENT_YEAR:
            debuts.append(d)
            fins.append(f)
    etendue = (max(fins) - min(debuts)) if debuts else 0
    return float(min(max(explicites + [etendue]), 45))


def annees_exigees_offre(texte: str) -> float | None:
    """Nombre minimal d'années demandé par l'offre (« 3+ years », « 5-7 years ») ; None si non précisé."""
    valeurs = []
    for m in _ANNEES_EXPLICITES.finditer(texte):
        suite = texte[m.end(): m.end() + 60].lower()
        # on ne garde que les vraies exigences (« 3+ years », « 5 years of experience »),
        # pas l'historique de l'entreprise (« for the past 20 years, we… »)
        if "+" in m.group(0) or re.search(r"\b(experience|exp)\b", suite):
            valeurs.append(_valeur(m.group(1)))
    valeurs = [v for v in valeurs if 0 < v <= 20]
    return float(min(valeurs)) if valeurs else None
