"""
ÉTAPE 0 — Téléchargement des données (Hugging Face) et du modèle d'embeddings, s'ils sont absents.

    python scripts/00_telecharger_donnees.py

Les fichiers ne sont pas versionnés dans Git (66 Mo de données, 88 Mo de modèle) : ce script les récupère
à une version figée (commit Hugging Face) pour garantir la reproductibilité.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recrutement import config as C

VERSION = "08978e21714984bb417547d2c0f9b477f5298163"
URL = "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/{v}/{f}"

C.RAW_DIR.mkdir(parents=True, exist_ok=True)
for fichier in ("train.csv", "test.csv"):
    cible = C.RAW_DIR / fichier
    if cible.exists():
        print(f"   {fichier} déjà présent")
        continue
    print(f"   téléchargement de {fichier}…")
    urllib.request.urlretrieve(URL.format(v=VERSION, f=fichier), cible)

modele = C.MODELS_DIR / "all-MiniLM-L6-v2"
if not (modele / "model.safetensors").exists():
    print("   téléchargement du modèle d'embeddings all-MiniLM-L6-v2 (88 Mo)…")
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # évite un blocage observé avec le protocole Xet
    from huggingface_hub import snapshot_download
    snapshot_download(C.EMBEDDING_HUB_ID, local_dir=str(modele),
                      allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"])
print("Données et modèle prêts.")
