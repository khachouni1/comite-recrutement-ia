"""
Démonstration interactive du comité multi-agents (Streamlit).

    streamlit run app/app.py          (après  python scripts/run_all.py)

Onglets : 1. Décision du comité   2. Fiche candidat   3. Nouveau candidat   4. Résultats ML & XAI
La barre latérale permet de saisir ses propres jugements AHP : le classement est recalculé en direct.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from recrutement import config as C
from recrutement.agents import EXAMINER, NON_RETENU, RECOMMANDE, VETO
from recrutement.mcdm import AHP

st.set_page_config(page_title="Comité de recrutement multi-agents", page_icon="👔", layout="wide")
ICONES = {RECOMMANDE: "✅", EXAMINER: "🟠", VETO: "⛔", NON_RETENU: "❌"}
LAB = C.CRITERIA_LABELS
DELIB_COLS = C.CRITERIA + [f"{p}_{c.replace('score_', '')}" for c in C.CRITERIA for p in ("confiance", "avis")] + \
    ["anomalie", "avis_anomalie", "ecart_avis", "desaccord", "veto", "contrefactuel_ok", "ecart_contrefactuel",
     "agents_sensibles"]


@st.cache_resource
def charger_comite():
    return joblib.load(C.MODELS_DIR / "comite.joblib")


@st.cache_data
def charger_classement():
    df = pd.read_csv(C.DATA_PROCESSED / "classement_test.csv", index_col="id")
    df[["veto", "agents_sensibles"]] = df[["veto", "agents_sensibles"]].fillna("")
    return df


if not (C.MODELS_DIR / "comite.joblib").exists():
    st.error("Comité non entraîné : exécutez d'abord  `python scripts/run_all.py`")
    st.stop()
comite, base = charger_comite(), charger_classement()

# ------------------------------------------------------------------ barre latérale : jugements AHP
st.sidebar.header("⚖️ Jugements AHP")
st.sidebar.caption("« Le critère de gauche est x fois plus important que celui de droite » (échelle de Saaty). "
                   "Poids = vecteur propre principal ; jugements acceptés si CR < 0.10.")
echelle = [1 / v for v in range(9, 1, -1)] + list(range(1, 10))
fmt = lambda v: f"1/{round(1 / v)}" if v < 1 else str(int(v))
defaut = np.array(C.AHP_MATRIX, dtype=float)
jugements = {}
for i, ci in enumerate(C.CRITERIA):
    for j, cj in enumerate(C.CRITERIA):
        if j > i:
            proche = min(echelle, key=lambda v: abs(np.log(v) - np.log(defaut[i, j])))
            jugements[(ci, cj)] = st.sidebar.select_slider(f"{LAB[ci]} vs {LAB[cj]}", echelle, proche, fmt)
ahp = AHP.from_judgments(C.CRITERIA, jugements)
if ahp.is_consistent:
    poids = ahp.weights
    st.sidebar.success(f"CR = {ahp.cr:.3f} < 0.10 : jugements cohérents")
else:
    poids = comite.poids
    st.sidebar.error(f"CR = {ahp.cr:.3f} ≥ 0.10 : jugements incohérents, poids par défaut conservés.")
for c in C.CRITERIA:
    st.sidebar.progress(float(poids[c]), text=f"{LAB[c]} : {poids[c]:.1%}")
places = st.sidebar.slider("Nombre de places (short-list)", 10, 120, comite.nb_places(len(base)))

# ------------------------------------------------------------------ décision recalculée en direct
dec = comite.decider(base[DELIB_COLS], poids, places).join(base[C.ALL_FEATURES + [C.TARGET]])
audit = comite.ethique.audit_selection(dec)
y = dec[C.TARGET]

st.title("👔 Comité multi-agents d'aide à la sélection de candidats")
st.caption("Sujet 11 — ECC, Ingénierie Machine Learning & Statistique. Données : 225 candidats du jeu de test.")
m = st.columns(4)
m[0].metric("Sélectionnés", int(dec["selectionne"].sum()), f"dont {int((dec['decision'] == EXAMINER).sum())} à examiner")
m[1].metric("Embauchés réels dans la sélection", f"{y[dec['selectionne']].mean():.0%}", f"hasard : {y.mean():.0%}")
m[2].metric("AUC du classement TOPSIS", f"{roc_auc_score(y, dec['TOPSIS_score']):.3f}",
            f"WSM : {roc_auc_score(y, dec['WSM_score']):.3f}")
m[3].metric("Audit éthique", "Conforme" if audit["conforme"] else f"{len(audit['alertes'])} alerte(s)",
            f"DI genre = {audit['disparate_impact']['Genre']['DI']:.2f}")

onglets = st.tabs(["🏆 Décision du comité", "🤖 Fiche candidat", "🧪 Nouveau candidat", "📊 Résultats ML & XAI"])

# ------------------------------------------------------------------ 1. décision
with onglets[0]:
    filtre = st.multiselect("Décisions affichées", list(ICONES), default=list(ICONES))
    vue = dec[dec["decision"].isin(filtre)]
    colonnes = ["decision", "TOPSIS_rang", "TOPSIS_score", "WSM_rang"] + C.CRITERIA + ["ecart_avis", "justification", C.TARGET]
    st.dataframe(vue[colonnes].rename(columns={**LAB, C.TARGET: "Embauché (réel)"}), height=420, width="stretch")
    st.download_button("📥 Télécharger (CSV)", dec.to_csv().encode("utf-8"), "decision_comite.csv", "text/csv")
    st.subheader("⚖️ Audit éthique de la sélection")
    a = st.columns(2)
    for col, (g, info) in zip(a, audit["disparate_impact"].items()):
        col.markdown(f"**{g}** — Disparate Impact = `{info['DI']:.2f}` "
                     f"{'✅ règle des 4/5 respectée' if info['conforme'] else '⚠️ règle des 4/5 non respectée'}")
        col.bar_chart(pd.Series(info["taux_selection"], name="Taux de sélection"))
    for alerte in audit["alertes"]:
        st.warning(alerte)

# ------------------------------------------------------------------ 2. fiche candidat
with onglets[1]:
    cid = st.selectbox("Candidat (id)", dec.index.tolist())
    r = dec.loc[cid]
    st.markdown(f"### {ICONES[r['decision']]} {r['decision']} — rang TOPSIS {int(r['TOPSIS_rang'])}")
    st.caption(r["justification"])
    g, d = st.columns([1, 1.4])
    with g:
        st.bar_chart(pd.Series({LAB[c]: r[c] for c in C.CRITERIA}, name="Note /100"), horizontal=True)
        st.write(f"Écart d'avis entre agents : **{r['ecart_avis']:.2f}** {'⚠️ désaccord' if r['desaccord'] else ''}")
        st.write(f"Décision réelle : **{'embauché' if r[C.TARGET] == 1 else 'non retenu'}**")
        st.dataframe(r[C.ALL_FEATURES].rename(C.LABELS).to_frame("valeur"), width="stretch")
    with d:
        for c in C.CRITERIA:
            court = c.replace("score_", "")
            st.info(f"**{LAB[c]}** (confiance {r[f'confiance_{court}']:.2f}) — {r[f'avis_{court}']}")
        st.write(f"🔎 {r['avis_anomalie']}")
        st.write("⚖️ " + ("Test contrefactuel réussi : genre et âge ne changent aucune note."
                          if r["contrefactuel_ok"] else f"Alerte contrefactuelle : {r['agents_sensibles']}"))
        if r["veto"]:
            st.error(f"⛔ Véto : {r['veto']}")

# ------------------------------------------------------------------ 3. nouveau candidat
with onglets[2]:
    with st.form("nouveau"):
        c1, c2, c3 = st.columns(3)
        cand = {
            "SkillScore": c1.slider("Test technique", 0, 100, 80),
            "InterviewScore": c1.slider("Entretien", 0, 100, 70),
            "PersonalityScore": c1.slider("Personnalité", 0, 100, 70),
            "EducationLevel": c2.selectbox("Niveau d'études (1 à 4)", [1, 2, 3, 4], index=2),
            "ExperienceYears": c2.slider("Années d'expérience", 0, 15, 6),
            "PreviousCompanies": c2.slider("Entreprises précédentes", 1, 5, 2),
            "RecruitmentStrategy": c3.selectbox("Stratégie de recrutement (1 à 3)", [1, 2, 3], index=1),
            "DistanceFromCompany": c3.slider("Distance (km)", 1.0, 51.0, 15.0),
            "Age": c3.slider("Âge (sert uniquement à l'audit)", 20, 50, 30),
            "Gender": c3.selectbox("Genre (sert uniquement à l'audit)", [0, 1],
                                   format_func=lambda v: "Femme" if v == 0 else "Homme"),
        }
        go = st.form_submit_button("🚀 Faire délibérer le comité")
    if go:
        res = comite.evaluer_candidat(cand, base, poids)
        st.markdown(f"## {ICONES[res['decision']]} {res['decision']}")
        st.caption(f"Rang TOPSIS {res['rang']} / {res['taille_vivier']} (short-list de {res['nb_places']} places) — "
                   f"écart d'avis {res['ecart_avis']:.2f}")
        cols = st.columns(4)
        for col, c in zip(cols, C.CRITERIA):
            col.metric(LAB[c], f"{res['avis'][c].note:.0f}/100", f"percentile {res['percentiles'][c]:.0%}")
        for c in C.CRITERIA:
            st.write(f"**{LAB[c]}** (confiance {res['avis'][c].confiance:.2f}) — {res['avis'][c].texte}")
        if res["veto"]:
            st.error("⛔ Véto : " + ", ".join(res["veto"]))
        (st.warning if res["anomalie"].details["anomalie"] else st.write)(f"🔎 {res['anomalie'].texte}")
        st.write(f"⚖️ {res['ethique'].texte}")

# ------------------------------------------------------------------ 4. figures
with onglets[3]:
    figures = sorted(C.FIGURES_DIR.glob("*.png"))
    choix = st.selectbox("Figure", [f.stem for f in figures])
    st.image(str(C.FIGURES_DIR / f"{choix}.png"), width="stretch")
