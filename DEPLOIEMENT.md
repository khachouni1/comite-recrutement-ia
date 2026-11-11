# Mettre le site en ligne (Streamlit Community Cloud)

Tout est prêt dans le dépôt : le comité entraîné (`models/comite.joblib`), les exemples de démonstration
(`data/exemples_demo.csv`), les dépendances du site (`app/requirements.txt`) et la configuration (`.streamlit/`).
Le modèle d'embeddings est téléchargé automatiquement depuis Hugging Face au premier démarrage.

## 1. Envoyer le projet sur GitHub

1. Sur https://github.com/new, créez un dépôt **vide** : par exemple `comite-recrutement-ia`, en **Public**, sans
   README, sans .gitignore et sans licence.
2. Dans un terminal ouvert dans le dossier du projet, lancez les commandes suivantes, en remplaçant `VOTRE-COMPTE` :

```bash
git remote add origin https://github.com/VOTRE-COMPTE/comite-recrutement-ia.git
```

```bash
git push -u origin master
```

```bash
git push --tags
```

Au premier `git push`, une fenêtre de connexion GitHub peut s'ouvrir : connectez-vous.

## 2. Créer l'application sur Streamlit Cloud

1. Allez sur https://share.streamlit.io, puis **Create app** → **Deploy a public app from GitHub**.
2. Remplissez :
   - **Repository** : `VOTRE-COMPTE/comite-recrutement-ia`
   - **Branch** : `master`
   - **Main file path** : `app/app.py`
   - **App URL** : au choix, par exemple `comite-recrutement-ecc`
3. Ouvrez **Advanced settings** et choisissez **Python 3.13** (obligatoire : c'est la version qui a servi à
   entraîner le comité).
4. Cliquez sur **Deploy**.

Le premier démarrage prend **5 à 10 minutes** : installation de PyTorch et des bibliothèques, puis téléchargement du
modèle (environ 90 Mo). Les visites suivantes sont rapides.

## 3. Mettre à jour le site

Chaque `git push` sur la branche `master` redéploie automatiquement le site.

Si vous ré-entraînez le comité (`python scripts/run_all.py`), commitez le nouveau `models/comite.joblib` et
`data/exemples_demo.csv`, puis faites un `git push`.

## Problèmes fréquents

| Symptôme | Cause probable | Solution |
|---|---|---|
| Erreur au chargement du comité (`ModuleNotFoundError`, `AttributeError`, `InconsistentVersionWarning`) | Version de Python ou des bibliothèques différente de celle de l'entraînement | Vérifier **Python 3.13** dans les paramètres de l'app (⋮ → Settings), ne pas modifier les versions de `app/requirements.txt` |
| « Your app has gone over its resource limits » | Mémoire dépassée | Redémarrer l'app (⋮ → Reboot) ; le modèle utilisé est déjà le plus léger possible |
| L'app « s'endort » | Inactivité de plusieurs jours (comportement normal de l'offre gratuite) | Cliquer sur « Yes, get this app back up » |
| Installation très longue ou en échec sur `torch` | Téléchargement de PyTorch | Relancer le déploiement (⋮ → Reboot) |

Pour la démonstration (vidéo, soutenance), ouvrez le site **quelques minutes avant** : le premier chargement après
une mise en veille prend environ une minute.
