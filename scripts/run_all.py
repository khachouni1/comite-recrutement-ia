"""
Exécute tout le pipeline dans l'ordre (≈ 2 minutes) :

    python scripts/run_all.py
"""
import runpy
import time
from pathlib import Path

ETAPES = ["01_eda.py", "02_non_supervise.py", "03_supervise.py", "04_comite_mcdm.py"]

debut = time.time()
for etape in ETAPES:
    print(f"\n>>> {etape}")
    runpy.run_path(str(Path(__file__).parent / etape), run_name="__main__")
print(f"\nPipeline complet terminé en {time.time() - debut:.0f} s. "
      "Lancez la démonstration :  streamlit run app/app.py")
