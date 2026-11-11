"""
Site web du comité de recrutement multi-agents.

    streamlit run app/app.py          (après  python scripts/run_all.py)

1. Évaluer un candidat      : une offre + un CV  ->  décision argumentée critère par critère
2. Classer des candidats    : une offre + plusieurs CV  ->  classement TOPSIS
3. Comment ça marche        : méthode, résultats, figures
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from recrutement import config as C
from recrutement.agents import EXAMINER, NON_RETENU, RECOMMANDE, VETO
from recrutement.lexiques import DEGREE_NAMES
from recrutement.mcdm import AHP
from recrutement.texte import lire_fichier

st.set_page_config(page_title="Comité de recrutement IA", page_icon="👔", layout="wide")
LAB = C.CRITERIA_LABELS
STYLE = {RECOMMANDE: ("✅", "#27ae60"), EXAMINER: ("🟠", "#f39c12"), NON_RETENU: ("❌", "#7f8c8d"),
         VETO: ("⛔", "#c0392b")}
ICONES = {"score_technique": "🛠️", "score_experience": "📈", "score_formation": "🎓",
          "score_soft_skills": "🤝", "score_adequation": "🎯", "score_predictif": "🤖"}


@st.cache_resource(show_spinner="Chargement du comité d'agents…")
def charger_comite():
    return joblib.load(C.MODELS_DIR / "comite.joblib")


@st.cache_data
def exemples() -> pd.DataFrame:
    return pd.read_csv(C.RAW_DIR / "test.csv")


if not (C.MODELS_DIR / "comite.joblib").exists():
    st.error("Le comité n'est pas encore entraîné : exécutez  `python scripts/run_all.py`.")
    st.stop()
comite = charger_comite()

# ---------------------------------------------------------------- barre latérale : priorités
st.sidebar.header("⚖️ Priorités du recruteur")
mode = st.sidebar.radio("Pondération des critères", ["Comité (AHP expert)", "Mes priorités"], index=0)
if mode == "Mes priorités":
    st.sidebar.caption("Importance de 1 (faible) à 9 (essentielle). Une matrice AHP cohérente est construite "
                       "avec a_ij = importance_i / importance_j.")
    prio = {c: st.sidebar.slider(LAB[c], 1, 9, 5) for c in C.CRITERIA}
    poids = AHP(C.CRITERIA, [[prio[i] / prio[j] for j in C.CRITERIA] for i in C.CRITERIA]).weights
else:
    poids = comite.poids
    st.sidebar.caption(f"Matrice de comparaison par paires du comité : CR = {comite.ahp.cr:.3f} < 0.10.")
for c in C.CRITERIA:
    st.sidebar.progress(float(poids[c]), text=f"{ICONES[c]} {LAB[c]} : {poids[c]:.0%}")
st.sidebar.divider()
st.sidebar.caption("⚠️ Le comité a été entraîné sur des CV et des offres **en anglais**. "
                   "Outil d'aide à la décision : la décision finale revient à un humain.")

st.title("👔 Comité de recrutement multi-agents")
st.caption("Six agents évaluent un CV face à une offre, un coordinateur agrège leurs avis (AHP + TOPSIS), "
           "deux agents de contrôle vérifient la qualité du dossier et l'équité.")
onglets = st.tabs(["🔎 Évaluer un candidat", "🏆 Classer plusieurs candidats", "📚 Comment ça marche"])


def zone_texte(titre: str, cle: str, hauteur: int = 260) -> str:
    fichier = st.file_uploader(f"{titre} — déposer un fichier (PDF, Word, TXT)", type=["pdf", "docx", "txt"],
                               key=f"f_{cle}")
    if fichier is not None:
        st.session_state[cle] = lire_fichier(fichier.name, fichier.getvalue())
    return st.text_area(f"{titre} — ou coller le texte", key=cle, height=hauteur)


def radar(notes: dict) -> go.Figure:
    noms = [LAB[c] for c in C.CRITERIA]
    vals = [notes[c] for c in C.CRITERIA]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=noms + noms[:1], fill="toself", name="Candidat"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=False, height=360,
                      margin=dict(l=40, r=40, t=30, b=30))
    return fig


# ================================================================= 1. évaluer un candidat
with onglets[0]:
    ex = exemples()
    with st.expander("Charger un exemple réel du jeu de test"):
        etiquette = st.selectbox("Exemple dont l'étiquette réelle est", C.LABELS,
                                 format_func=lambda l: C.LABELS_FR[l])
        if st.button("Charger cet exemple"):
            r = ex[ex["label"] == etiquette].sample(1).iloc[0]
            st.session_state["offre"], st.session_state["cv"] = r["job_description_text"], r["resume_text"]
            st.session_state["etiquette_exemple"] = etiquette
    g, d = st.columns(2)
    with g:
        offre = zone_texte("📄 Offre d'emploi", "offre")
    with d:
        cv = zone_texte("👤 CV du candidat", "cv")

    if st.button("🚀 Lancer l'évaluation du comité", type="primary", disabled=not (offre.strip() and cv.strip())):
        with st.spinner("Les agents délibèrent…"):
            res = comite.evaluer(cv, offre, poids)
        icone, couleur = STYLE[res["decision"]]
        st.markdown(f"<div style='padding:16px;border-radius:10px;background:{couleur}22;border-left:6px solid {couleur}'>"
                    f"<h2 style='margin:0'>{icone} {res['decision']}</h2>"
                    f"<p style='margin:4px 0 0 0'>{res['justification']}</p></div>", unsafe_allow_html=True)
        if st.session_state.get("etiquette_exemple"):
            st.caption(f"Étiquette réelle de cet exemple : **{C.LABELS_FR[st.session_state['etiquette_exemple']]}**")

        m = st.columns(4)
        m[0].metric("Score global (TOPSIS)", f"{res['score_global']:.0f}/100")
        m[1].metric("Baseline (somme pondérée)", f"{res['score_wsm']:.0f}/100")
        m[2].metric("Avis prédictif (ML)", f"{res['predictif'].note:.0f}/100")
        m[3].metric("Seuil de recommandation", f"{comite.seuil_reco:.0f}/100")

        col_radar, col_avis = st.columns([1, 1.5])
        with col_radar:
            st.plotly_chart(radar({c: a.note for c, a in res["avis"].items()}), width="stretch")
            det = res["details"]
            exige = "—" if det["annees_exigees"] is None else f"{det['annees_exigees']:.0f}"
            st.markdown(f"**Extrait du texte** — expérience : {det['annees_cv']:.0f} an(s) (exigé : {exige}) · "
                        f"diplôme : {DEGREE_NAMES[det['diplome_cv']]} (exigé : {DEGREE_NAMES[det['diplome_exige']]})")
        with col_avis:
            for c, a in res["avis"].items():
                with st.container(border=True):
                    st.markdown(f"**{ICONES[c]} {a.agent} — {a.note:.0f}/100** · confiance {a.confiance:.0%}")
                    st.write(a.texte)
                    if c in ("score_technique", "score_soft_skills"):
                        if a.points_forts:
                            st.markdown("✅ " + " · ".join(f"`{s}`" for s in a.points_forts))
                        if a.points_faibles:
                            st.markdown("❌ " + " · ".join(f"`{s}`" for s in a.points_faibles))
        st.markdown("#### Agents de contrôle")
        k2, k3 = st.columns(2)
        (k2.warning if res["anomalie"].details["anomalie"] else k2.success)(f"🔎 **{res['anomalie'].agent}** — "
                                                                             f"{res['anomalie'].texte}")
        cf = res["ethique"].details.get("contrefactuel") or {"ok": True}
        (k3.success if cf["ok"] else k3.error)(f"⚖️ **{res['ethique'].agent}** — {res['ethique'].texte}")
        if res["veto"]:
            st.error(f"⛔ Critère éliminatoire non atteint : {res['veto']}")

# ================================================================= 2. classer plusieurs candidats
with onglets[1]:
    st.markdown("Une offre, plusieurs CV : le comité note chaque candidat, TOPSIS les classe entre eux.")
    ex = exemples()
    if st.button("Charger une offre réelle et ses candidats (jeu de test)"):
        offre_ex = ex["job_description_text"].value_counts().index[0]
        sous = ex[ex["job_description_text"] == offre_ex].head(12)
        st.session_state["offre_multi"] = offre_ex
        st.session_state["cvs_multi"] = {f"Candidat {i + 1}": t for i, t in enumerate(sous["resume_text"])}
        st.session_state["verite_multi"] = {f"Candidat {i + 1}": C.LABELS_FR[l] for i, l in enumerate(sous["label"])}
    offre_m = zone_texte("📄 Offre d'emploi", "offre_multi", 200)
    fichiers = st.file_uploader("👥 Déposer plusieurs CV (PDF, Word, TXT)", type=["pdf", "docx", "txt"],
                                accept_multiple_files=True, key="cvs_files")
    cvs = dict(st.session_state.get("cvs_multi", {}))
    for f in fichiers or []:
        cvs[f.name] = lire_fichier(f.name, f.getvalue())
    st.caption(f"{len(cvs)} CV prêt(s).")
    if st.button("🏆 Classer les candidats", type="primary", disabled=not (offre_m.strip() and len(cvs) >= 2)):
        with st.spinner("Évaluation de chaque candidat…"):
            classement = comite.classer(cvs, offre_m, poids)
        vue = classement[["rang", "decision", "score_global"] + C.CRITERIA].rename(
            columns={**LAB, "score_global": "Score global", "decision": "Décision", "rang": "Rang"})
        if "verite_multi" in st.session_state:
            vue["Étiquette réelle"] = vue.index.map(st.session_state["verite_multi"])
        st.dataframe(vue.style.format(precision=0).background_gradient(subset=[LAB[c] for c in C.CRITERIA],
                                                                       cmap="Greens", vmin=0, vmax=100),
                     width="stretch")
        st.bar_chart(classement["score_global"].rename("Score global"))

# ================================================================= 3. comment ça marche
with onglets[2]:
    st.markdown("""
**Démarche** (code dans `src/recrutement`, une étape par script dans `scripts/`) :
1. **Données** : 8 000 paires CV / offre étiquetées *adapté, potentiel, non adapté* (Hugging Face).
   Découpage Train / Validation / Test **par offre** : le Test ne contient que des postes jamais vus.
2. **Texte** : nettoyage, **anonymisation**, extraction de critères (compétences, années, diplômes, soft skills),
   représentations TF-IDF → LSA (ACP du texte) → embeddings Hugging Face ; clustering des CV en familles de métiers.
3. **Apprentissage supervisé** : régression softmax (baseline) vs arbres, forêts, gradient boosting, XGBoost ;
   SMOTE, validation croisée groupée, SHAP, permutation, analyse des erreurs.
4. **Comité** : 6 agents appris sur les données, poids **AHP** (CR < 0.10), score **TOPSIS** comparé à la baseline
   **WSM**, véto, second avis ML, contrôle des anomalies, audit éthique.
""")
    synth = C.RESULTS_DIR / "04_synthese.json"
    if synth.exists():
        ev = pd.DataFrame(json.loads(synth.read_text(encoding="utf-8"))["evaluation"]).T
        st.markdown("**Résultats sur le Test**")
        st.dataframe(ev.style.format("{:.3f}"), width="stretch")
    figures = sorted(C.FIGURES_DIR.glob("*.png"))
    if figures:
        choix = st.selectbox("Figure", [f.stem for f in figures])
        st.image(str(C.FIGURES_DIR / f"{choix}.png"), width="stretch")
