"""
Package « recrutement » — Sujet 11 (ECC) :
Système multi-agents fondé sur les méthodes MCDM pour l'aide à la sélection de candidats.

Modules, dans l'ordre du pipeline :
    config          constantes et chemins
    data            chargement + découpage Train / Val / Test
    preprocessing   pipeline de prétraitement (standardisation, one-hot, SMOTE)
    eda             analyse exploratoire
    unsupervised    ACP, clustering, détection d'anomalies
    supervised      modèles supervisés, sélection, évaluation
    explain         interprétabilité (SHAP, permutation, erreurs)
    mcdm/           AHP, WSM, TOPSIS, sensibilité
    agents/         agents évaluateurs, agents de contrôle, coordinateur
"""
