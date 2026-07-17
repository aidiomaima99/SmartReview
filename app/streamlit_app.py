"""
SmartReview Lite — Application Streamlit
Master BD & IA | AIDI Omaima | ELMOUSSAOUI Fatima
"""

from pathlib import Path
import sys

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import preprocess_full  # noqa: E402

MODEL_PATH = ROOT / "models" / "best_pipeline.joblib"


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="SmartReview Lite", page_icon="🎬", layout="centered")
    st.title("SmartReview Lite")
    st.caption("Analyse automatique des sentiments d'avis clients — Master BD & IA")

    pipeline = load_pipeline()
    if pipeline is None:
        st.error(
            "Modèle introuvable. Exécutez d'abord : `python3 src/train_evaluate.py`"
        )
        return

    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]
    model_name = pipeline.get("model_name", "Modèle")

    st.info(f"Modèle chargé : **{model_name}** (TF-IDF + prétraitement NLP)")

    review = st.text_area(
        "Saisissez un avis client :",
        height=160,
        placeholder="Exemple : This movie was absolutely fantastic, the acting was superb!",
    )

    if st.button("Analyser le sentiment", type="primary"):
        if not review.strip():
            st.warning("Veuillez saisir un avis.")
            return
        cleaned = preprocess_full(review)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        label = "Positif" if pred == 1 else "Négatif"
        color = "green" if pred == 1 else "red"

        # Probabilité si disponible
        proba_txt = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf = float(max(proba)) * 100
            proba_txt = f" (confiance : {conf:.1f}%)"
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
            proba_txt = f" (score décision : {score:.3f})"

        st.markdown(
            f"### Sentiment prédit : :{color}[**{label}**]{proba_txt}"
        )
        with st.expander("Texte après prétraitement NLP"):
            st.code(cleaned or "(vide après nettoyage)")

    st.divider()
    st.markdown(
        """
        **Projet SmartReview** — Benchmark NB / LR / SVM / RF sur IMDb 50K  
        Réalisé par AIDI Omaima & ELMOUSSAOUI Fatima — Encadrée par Prof. M. EL HAJJI — 2025-2026
        """
    )


if __name__ == "__main__":
    main()
