"""
Exécute tout le pipeline dans l'ordre :

    python scripts/run_all.py
"""
import runpy
import time
from pathlib import Path

ETAPES = ["00_telecharger_donnees.py", "01_donnees_eda.py", "02_representation_non_supervise.py", "03_supervise.py", "04_comite_mcdm.py"]

debut = time.time()
for etape in ETAPES:
    print(f"\n>>> {etape}")
    runpy.run_path(str(Path(__file__).parent / etape), run_name="__main__")
print(f"\nPipeline complet terminé en {time.time() - debut:.0f} s. Lancez le site :  streamlit run app/app.py")
