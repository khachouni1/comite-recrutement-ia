"""Tests de l'extraction de critères et de l'anonymisation (règles explicables)."""
from recrutement import features as F
from recrutement import texte as T


def test_nettoyage_separe_les_mots_colles():
    assert T.nettoyer("SummaryCertified  Electrical\nForeman") == "Summary Certified Electrical Foreman"


def test_anonymisation_retire_les_informations_protegees():
    brut = ("Mrs. Jane Doe, jane.doe@mail.com, +1 (555) 123-4567. She is 45 years old, married. "
            "Date of birth: 1980. Nationality: French. Python developer.")
    propre = T.anonymiser(brut).lower()
    for interdit in ["mrs", "@", "555", "she", "45 years old", "married", "1980", "french"]:
        assert interdit not in propre, interdit
    assert "python developer" in propre  # le contenu professionnel est conservé


def test_competences():
    cv = "Experienced in Python, SQL and Power BI dashboards. Strong communication and teamwork."
    assert {"python", "sql", "power bi", "data visualization"} <= T.competences(cv, "hard")
    assert {"communication", "teamwork"} <= T.competences(cv, "soft")
    assert "java" not in T.competences("I use JavaScript", "hard")  # pas de faux positif java/javascript


def test_annees_experience():
    assert T.annees_experience_cv("Accountant with over twelve (12) years of experience.") == 12
    assert T.annees_experience_cv("Analyst, Company A, 2015 - 2019. Lead, Company B, 2019 - 2023.") == 8
    assert T.annees_exigees_offre("Requirements: 3+ years of experience with Azure.") == 3
    assert T.annees_exigees_offre("For the past 20 years, we have powered digital experiences.") is None


def test_niveau_diplome():
    assert T.niveau_diplome("Master of Science in Finance; Bachelor of Arts") == 4
    assert T.niveau_diplome("Bachelor's degree required, Master's preferred", exigence=True) == 3
    assert T.niveau_diplome("No formal education listed") == 0


def test_comparer_couverture():
    a_cv = F.analyser_cv("Python and SQL developer, 5 years of experience, Bachelor of Science.")
    a_of = F.analyser_offre("We need Python, SQL and AWS. 3+ years of experience. Bachelor's degree.")
    f = F.comparer(a_cv, a_of)
    assert f["couverture_hard"] == 2 / 3 and f["ratio_annees"] == 5 / 3 and f["ecart_diplome"] == 0
    d = F.details(a_cv, a_of)
    assert d["hard_manquantes"] == ["aws"]
