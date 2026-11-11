"""Style graphique commun et sauvegarde des figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11})

PALETTE_LABELS = {"No Fit": "#c0392b", "Potential Fit": "#f39c12", "Good Fit": "#27ae60"}
DECISION_COLORS = {"Recommandé": "#27ae60", "À examiner": "#f39c12", "Non retenu": "#95a5a6",
                   "Écarté (véto)": "#c0392b"}


def save(fig, name: str) -> None:
    """Enregistre la figure dans reports/figures/<name>.png et la ferme."""
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"   figure -> reports/figures/{name}.png")
