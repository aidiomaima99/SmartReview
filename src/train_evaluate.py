"""
SmartReview — Step 5 : Expérimentation complète
Master BD & IA | AIDI Omaima | ELMOUSSAOUI Fatima | Encadrée par Prof. M. EL HAJJI

Protocole (Step 4) :
  - Dataset IMDb 50K (Kaggle / Stanford ACL)
  - TF-IDF max_features=5000, ngram_range=(1,2)
  - Split 80/20, random_state=42
  - Modèles : Naive Bayes, Logistic Regression, SVM, Random Forest
  - Conditions : avec / sans prétraitement NLP (H2)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import preprocess_full, preprocess_minimal  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)


def load_imdb() -> pd.DataFrame:
    """Charge IMDb depuis CSV Kaggle ou dossier ACL Stanford."""
    candidates = [
        DATA_DIR / "IMDB Dataset.csv",
        DATA_DIR / "imdb_dataset.csv",
        DATA_DIR / "IMDb_Dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            review_col = cols.get("review") or cols.get("text") or list(df.columns)[0]
            sentiment_col = cols.get("sentiment") or cols.get("label") or list(df.columns)[1]
            out = pd.DataFrame(
                {
                    "review": df[review_col].astype(str),
                    "sentiment": df[sentiment_col]
                    .astype(str)
                    .str.lower()
                    .map({"positive": 1, "negative": 0, "pos": 1, "neg": 0, "1": 1, "0": 0}),
                }
            )
            out = out.dropna().reset_index(drop=True)
            print(f"[OK] Dataset chargé depuis {path.name} — {len(out)} avis")
            return out

    acl = DATA_DIR / "aclImdb"
    if acl.exists():
        rows = []
        for split in ("train", "test"):
            for label_name, label in (("pos", 1), ("neg", 0)):
                folder = acl / split / label_name
                for fp in folder.glob("*.txt"):
                    rows.append({"review": fp.read_text(encoding="utf-8", errors="ignore"), "sentiment": label})
        out = pd.DataFrame(rows)
        print(f"[OK] Dataset ACL IMDb chargé — {len(out)} avis")
        return out

    raise FileNotFoundError(
        "Dataset introuvable. Placez 'IMDB Dataset.csv' dans data/ "
        "ou extrayez aclImdb_v1.tar.gz dans data/aclImdb/"
    )


def build_models() -> dict:
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver="liblinear"
        ),
        "SVM (LinearSVC)": LinearSVC(random_state=RANDOM_STATE, max_iter=5000, dual=True),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=40
        ),
    }


def evaluate_condition(df: pd.DataFrame, use_full_preprocess: bool, tag: str) -> pd.DataFrame:
    print(f"\n===== Condition : {tag} =====")
    preprocess_fn = preprocess_full if use_full_preprocess else preprocess_minimal

    print("Prétraitement en cours...")
    texts = df["review"].apply(preprocess_fn)
    y = df["sentiment"].values

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    t0 = time.perf_counter()
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
    vec_time = time.perf_counter() - t0
    print(f"TF-IDF : {X_train.shape} (vectorisation {vec_time:.2f}s)")

    rows = []
    reports = {}
    cms = {}

    for name, model in build_models().items():
        print(f"  → Entraînement {name}...")
        t_train0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t_train0

        t_pred0 = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = time.perf_counter() - t_pred0

        # CV 3-fold (échantillon pour RF afin de rester tractable)
        if name == "Random Forest":
            # Sous-échantillon pour CV uniquement (RF reste évalué sur le test complet)
            from sklearn.utils import resample

            idx = resample(
                range(X_train.shape[0]),
                n_samples=min(12000, X_train.shape[0]),
                random_state=RANDOM_STATE,
                replace=False,
            )
            cv_scores = cross_val_score(
                model, X_train[idx], y_train[idx], cv=3, scoring="f1", n_jobs=-1
            )
        else:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1
            )

        metrics = {
            "condition": tag,
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "train_time_s": float(train_time),
            "predict_time_s": float(pred_time),
            "total_time_s": float(train_time + pred_time),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
        }
        rows.append(metrics)
        reports[name] = classification_report(
            y_test, y_pred, target_names=["negative", "positive"], digits=4
        )
        cms[name] = confusion_matrix(y_test, y_pred).tolist()

        print(
            f"     Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
            f"time={metrics['total_time_s']:.2f}s  CV-F1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}"
        )

        # Sauvegarder le meilleur pipeline (condition avec prétraitement)
        if use_full_preprocess:
            joblib.dump(
                {"model": model, "vectorizer": vectorizer, "preprocess": "full"},
                MODELS_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib",
            )

    # Sauvegarder vectorizer + meilleur modèle pour Streamlit
    if use_full_preprocess:
        results_tmp = pd.DataFrame(rows)
        best_name = results_tmp.loc[results_tmp["f1"].idxmax(), "model"]
        best_model = build_models()[best_name]
        best_model.fit(X_train, y_train)
        joblib.dump(
            {
                "model": best_model,
                "vectorizer": vectorizer,
                "model_name": best_name,
                "preprocess": "full",
            },
            MODELS_DIR / "best_pipeline.joblib",
        )
        print(f"[OK] Meilleur modèle sauvegardé : {best_name}")

    # Figures matrices de confusion
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            np.array(cm),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
    plt.suptitle(f"Matrices de confusion — {tag}")
    plt.tight_layout()
    safe = tag.replace(" ", "_").lower()
    fig_path = FIG_DIR / f"confusion_matrices_{safe}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure sauvegardée : {fig_path.name}")

    with open(RESULTS_DIR / f"classification_reports_{safe}.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"confusion_matrices_{safe}.json", "w", encoding="utf-8") as f:
        json.dump(cms, f, indent=2)

    return pd.DataFrame(rows)


def plot_comparisons(df_all: pd.DataFrame) -> None:
    # Barplot métriques — avec prétraitement
    with_prep = df_all[df_all["condition"] == "Avec prétraitement NLP"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(with_prep))
    width = 0.18
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, with_prep[m].values, width, label=m.capitalize())
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(with_prep["model"].values, rotation=15)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des métriques (avec prétraitement NLP)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Temps d'exécution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(with_prep["model"], with_prep["total_time_s"], color="#2c7bb6")
    ax.set_xlabel("Temps total (s) — entraînement + prédiction")
    ax.set_title("Temps d'exécution par modèle (H3)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "execution_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Impact prétraitement (H2)
    pivot = df_all.pivot(index="model", columns="condition", values="f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, color=["#d7191c", "#1a9641"])
    ax.set_ylabel("F1-score")
    ax.set_title("Impact du prétraitement NLP sur le F1-score (H2)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(title="")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "preprocess_impact.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Compromis performance / temps (H3)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(with_prep["total_time_s"], with_prep["f1"], s=120, c="#2c7bb6")
    for _, r in with_prep.iterrows():
        ax.annotate(r["model"], (r["total_time_s"], r["f1"]), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Temps total (s)")
    ax.set_ylabel("F1-score")
    ax.set_title("Compromis performance / temps d'exécution (H3)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "performance_time_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Figures comparatives générées")


def validate_hypotheses(df_all: pd.DataFrame) -> dict:
    with_prep = df_all[df_all["condition"] == "Avec prétraitement NLP"].set_index("model")
    without = df_all[df_all["condition"] == "Sans prétraitement NLP"].set_index("model")

    # H1 : SVM meilleur F1
    best = with_prep["f1"].idxmax()
    h1 = {
        "statement": "H1 — SVM surpasse NB, RF et LR en F1-score",
        "best_model": best,
        "f1_scores": with_prep["f1"].to_dict(),
        "confirmed": bool(best.startswith("SVM")),
        "comment": (
            "Confirmée : SVM obtient le meilleur F1."
            if best.startswith("SVM")
            else f"Non confirmée telle quelle : le meilleur modèle est {best} (F1={with_prep.loc[best, 'f1']:.4f})."
        ),
    }

    # H2 : prétraitement améliore
    deltas = (with_prep["f1"] - without["f1"]).to_dict()
    improved = sum(1 for d in deltas.values() if d > 0)
    h2 = {
        "statement": "H2 — Le prétraitement NLP améliore les performances",
        "f1_deltas": deltas,
        "models_improved": improved,
        "confirmed": improved >= 3,
        "comment": (
            f"Confirmée : {improved}/4 modèles améliorent leur F1 avec prétraitement."
            if improved >= 3
            else f"Partiellement confirmée : {improved}/4 modèles améliorés."
        ),
    }

    # H3 : LR meilleur compromis
    # Score composite = F1 normalisé - temps normalisé
    f1_n = (with_prep["f1"] - with_prep["f1"].min()) / (with_prep["f1"].max() - with_prep["f1"].min() + 1e-9)
    t_n = (with_prep["total_time_s"] - with_prep["total_time_s"].min()) / (
        with_prep["total_time_s"].max() - with_prep["total_time_s"].min() + 1e-9
    )
    compromise = f1_n - t_n
    best_compromise = compromise.idxmax()
    h3 = {
        "statement": "H3 — La Régression Logistique offre le meilleur compromis performance/temps/simplicité",
        "compromise_scores": compromise.to_dict(),
        "best_compromise": best_compromise,
        "confirmed": bool(best_compromise == "Logistic Regression"),
        "comment": (
            "Confirmée : LR maximise le compromis F1 / temps."
            if best_compromise == "Logistic Regression"
            else f"Nuancée : le meilleur compromis mesuré est {best_compromise}."
        ),
    }

    return {"H1": h1, "H2": h2, "H3": h3}


def main() -> None:
    print("=" * 60)
    print("SmartReview — Expérimentation Master BD & IA")
    print("=" * 60)

    df = load_imdb()
    print("\nDistribution des classes :")
    print(df["sentiment"].value_counts().rename({0: "negative", 1: "positive"}))
    print(f"Longueur moyenne des avis : {df['review'].str.len().mean():.0f} caractères")

    # EDA rapide
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = df["sentiment"].map({0: "Negative", 1: "Positive"})
    labels.value_counts().plot(kind="bar", ax=axes[0], color=["#d7191c", "#1a9641"])
    axes[0].set_title("Distribution des classes (IMDb 50K)")
    axes[0].set_ylabel("Nombre d'avis")
    axes[0].tick_params(axis="x", rotation=0)
    df["review"].str.len().hist(bins=50, ax=axes[1], color="#2c7bb6")
    axes[1].set_title("Distribution de la longueur des avis")
    axes[1].set_xlabel("Caractères")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Deux conditions expérimentales (H2)
    res_with = evaluate_condition(df, use_full_preprocess=True, tag="Avec prétraitement NLP")
    res_without = evaluate_condition(df, use_full_preprocess=False, tag="Sans prétraitement NLP")

    df_all = pd.concat([res_with, res_without], ignore_index=True)
    df_all.to_csv(RESULTS_DIR / "metrics_all.csv", index=False)
    df_all.to_json(RESULTS_DIR / "metrics_all.json", orient="records", indent=2)

    plot_comparisons(df_all)
    hypotheses = validate_hypotheses(df_all)
    with open(RESULTS_DIR / "hypotheses_validation.json", "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2, ensure_ascii=False)

    print("\n===== Validation des hypothèses =====")
    for key, h in hypotheses.items():
        status = "CONFIRMÉE" if h["confirmed"] else "NON CONFIRMÉE / NUANCÉE"
        print(f"{key}: {status} — {h['comment']}")

    print("\n[OK] Expérimentation terminée. Résultats dans results/")


if __name__ == "__main__":
    main()
