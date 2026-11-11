# Données

## Source

- **Jeu de données :** *resume-job-description-fit* (auteur : cnamuangtoun), Hugging Face
- **Lien :** https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit
- **Version utilisée :** commit `08978e21714984bb417547d2c0f9b477f5298163`
- **Licence :** aucune licence n'est déclarée sur la page du jeu de données. Il est utilisé ici dans un cadre
  pédagogique et non commercial ; à mentionner dans le rapport.
- **Fichiers :** `raw/cv_offres/train.csv` (6 241 paires) et `raw/cv_offres/test.csv` (1 759 paires)

Pour retélécharger les fichiers :

```bash
curl -L -o data/raw/cv_offres/train.csv https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/08978e21714984bb417547d2c0f9b477f5298163/train.csv
curl -L -o data/raw/cv_offres/test.csv https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/08978e21714984bb417547d2c0f9b477f5298163/test.csv
```

## Contenu

| Colonne | Description |
|---|---|
| `resume_text` | Texte du CV (anglais) : environ 620 mots en médiane |
| `job_description_text` | Texte de l'offre d'emploi (anglais) : environ 320 mots en médiane |
| `label` | **Cible** : `No Fit` (≈ 50 %), `Potential Fit` (≈ 25 %), `Good Fit` (≈ 25 %) |

Il y a seulement **643 CV et 351 offres distincts** : chaque CV est associé à plusieurs offres. Les métiers
couverts sont l'informatique, la data, l'analyse métier, la comptabilité et la finance, la gestion de projet,
la santé et l'ingénierie.

## Traitements effectués

1. **Doublons** : une paire en double est retirée.
2. **Nettoyage** : séparation des mots collés (« SummaryCertified » devient « Summary Certified ») et normalisation des espaces.
3. **Anonymisation** (`src/recrutement/texte.py`) : on retire les e-mails, téléphones, liens, dates de naissance, âges,
   situations familiales, civilités, pronoms genrés, mentions de genre, de nationalité et de religion. **Limite :** les noms
   propres ne sont pas retirés, car il faudrait un modèle de reconnaissance d'entités nommées.
4. **Découpage** : le Test est le test officiel (71 offres). Le train officiel est découpé **par offre** en Train (85 %)
   et Validation (15 %). Aucune offre n'est partagée entre les trois jeux.
5. **Extraction de critères** par règles et lexiques : compétences, soft skills, années d'expérience, niveau de diplôme.
6. **Représentations** apprises sur le Train : TF-IDF, LSA et embeddings Hugging Face (`all-MiniLM-L6-v2`).

Le dossier `processed/` est régénéré par `python scripts/run_all.py`.
