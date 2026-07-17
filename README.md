# SmartReview

**Analyse automatique des sentiments des avis clients par Machine Learning**

| | |
|---|---|
| **Niveau** | Master BD & IA — Méthodologie de recherche |
| **Auteures** | AIDI Omaima · ELMOUSSAOUI Fatima |
| **Encadrement** | Prof. M. EL HAJJI |
| **Année** | 2025–2026 |
| **Dataset** | [IMDb 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (équivalent Kaggle) |
| **Stack** | Python · scikit-learn · TF-IDF · Streamlit |

---

## En une phrase

SmartReview compare **4 algorithmes de Machine Learning** pour classer des avis de films en **positif** ou **négatif**, puis propose une petite application web pour tester un avis en direct.

---

## Pourquoi ce projet ?

Sur Internet, les clients écrivent des millions d’avis. Lire tous ces textes à la main est impossible.  
On veut donc un **ordinateur capable de comprendre** si un avis est positif ou négatif.

**Question de recherche :**  
Parmi Naive Bayes, Régression Logistique, SVM et Random Forest, **quel modèle est le plus performant** (et le plus pratique) pour cette tâche ?

---

## Comment ça marche ? (explication débutant)

Imagine une chaîne de montage en **6 étapes** :

```
Avis brut  →  Nettoyage  →  Transformation en nombres  →  Apprentissage  →  Note  →  Application
 (texte)      (NLP)         (TF-IDF)                     (4 modèles)     (métriques)  (Streamlit)
```

### Étape A — Les données (le carburant)

On utilise **50 000 avis** IMDb :
- **25 000** positifs  
- **25 000** négatifs  

Chaque avis a déjà une étiquette (comme une correction d’examen).  
Le modèle va **apprendre** sur 80 % des avis, puis on le **teste** sur les 20 % restants.

### Étape B — Le nettoyage du texte (prétraitement NLP)

Un avis brut contient du “bruit” (balises HTML, ponctuation, mots inutiles).

Exemple :

| Avant | Après |
|-------|--------|
| `This movie was <br> absolutely FANTASTIC!!!` | `movie absolutely fantastic` |

On fait :
1. minuscules  
2. suppression HTML / ponctuation  
3. suppression des mots vides (*the*, *and*, *is*…)  
4. lemmatisation légère (*movies* → *movie*)

### Étape C — Transformer le texte en nombres (TF-IDF)

Un modèle ML ne lit pas du français/anglais : il lit des **nombres**.

**TF-IDF** donne un score à chaque mot :
- fréquent dans **un** avis → important  
- fréquent dans **tous** les avis → peu utile  

Résultat : chaque avis devient un vecteur de **5 000** valeurs (unigrammes + bigrammes).

### Étape D — Les 4 “élèves” (modèles)

| Modèle | Idée simple |
|--------|-------------|
| **Naive Bayes** | Probabilités de mots “positifs” vs “négatifs” |
| **Régression Logistique** | Trace une frontière linéaire entre + et − |
| **SVM** | Cherche la “meilleure marge” entre les 2 classes |
| **Random Forest** | Vote de beaucoup d’arbres de décision |

Tous apprennent sur **les mêmes données**, pour un comparatif juste.

### Étape E — La notation (métriques)

| Métrique | Signification débutant |
|----------|-------------------------|
| **Accuracy** | % de bonnes réponses au total |
| **Precision** | Quand je dis “positif”, ai-je raison ? |
| **Recall** | Ai-je trouvé tous les vrais positifs ? |
| **F1-score** | Équilibre Precision / Recall (métrique principale) |
| **Temps** | Vitesse d’apprentissage + prédiction |

### Étape F — L’application (preuve de concept)

Une interface **Streamlit** : tu colles un avis → le modèle répond **Positif** ou **Négatif**.

---

## Méthodologie de recherche (Steps 1 → 7)

Ce projet suit un **cycle de recherche complet** (pas seulement du code) :

| Step | Titre | Rôle |
|------|-------|------|
| **1** | Problématique | Définir le problème et la question de recherche |
| **2** | Revue de littérature | Lire l’état de l’art (NLP, sentiment analysis) |
| **3** | Hypothèses | Formuler H1, H2, H3 |
| **4** | Plan expérimental | Protocole, variables, outils, validité |
| **5** | Collecte / expérimentation | Télécharger les données, entraîner, mesurer |
| **6** | Analyse / interprétation | Valider ou nuancer les hypothèses |
| **7** | Publication / dissémination | Rapports, abstract, perspectives |

### Nos 3 hypothèses

| ID | Hypothèse | Résultat obtenu |
|----|-----------|-----------------|
| **H1** | SVM bat NB, LR et RF en F1 | **Nuancée** — LR légèrement devant (F1 = 0.889) |
| **H2** | Le prétraitement NLP améliore tous les modèles | **Partielle** — 2 modèles sur 4 s’améliorent |
| **H3** | La Régression Logistique offre le meilleur compromis performance / temps / simplicité | **Confirmée** |

> En recherche, **réfuter ou nuancer** une hypothèse est aussi valide que la confirmer : cela montre une analyse critique.

---

## Résultats principaux

Condition : **avec prétraitement NLP** · split 80/20 · `random_state=42`

| Modèle | Accuracy | Precision | Recall | F1 | Temps total |
|--------|----------|-----------|--------|-----|-------------|
| **Logistic Regression** | **88.71 %** | 0.875 | 0.904 | **0.889** | **1.8 s** |
| SVM (LinearSVC) | 88.39 % | 0.875 | 0.895 | 0.885 | 1.2 s |
| Naive Bayes | 85.98 % | 0.846 | 0.880 | 0.863 | 0.05 s |
| Random Forest | 85.15 % | 0.851 | 0.853 | 0.852 | 77 s |

**Verdict pratique :**  
→ **Régression Logistique** = meilleur équilibre précision / rapidité pour un déploiement léger (*SmartReview Lite*).

Benchmark littérature (Puh & Bagic Babac, 2023) : SVM ≈ 80 %, NB ≈ 73 % sur un autre corpus d’avis.  
Nos résultats ML classiques sur IMDb sont **cohérents et compétitifs**.

---

## Structure du dépôt

```
SmartReview/
├── README.md                 ← ce fichier
├── requirements.txt          ← dépendances Python
├── app/
│   └── streamlit_app.py      ← démo interactive
├── src/
│   ├── preprocess.py         ← nettoyage NLP
│   ├── train_evaluate.py     ← expérience complète (Steps 5–6)
│   └── generate_reports.py   ← génération des PDF
├── notebooks/
│   └── SmartReview_Experimentation.ipynb
├── data/
│   └── aclImdb/              ← dataset (à télécharger, voir ci-dessous)
├── models/                   ← modèles sauvegardés (.joblib)
├── results/
│   ├── metrics_all.csv       ← tableau de résultats
│   ├── hypotheses_validation.json
│   └── figures/              ← graphiques
└── reports/
    ├── Step5_Collecte_Experimentation.pdf
    ├── Step6_Interpretation_Analyse.pdf
    └── Step7_Publication_Dissemination.pdf
```

---

## Installation rapide

### 1. Cloner / ouvrir le projet

```bash
cd SmartReview   
```

### 2. Installer les dépendances

```bash
pip3 install -r requirements.txt
```

### 3. Télécharger le dataset

**Option A — Stanford (recommandée, sans compte) :**

```bash
mkdir -p data && cd data
curl -L -o aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
cd ..
```

**Option B — Kaggle :**  
Dataset : [`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
Placer le fichier `IMDB Dataset.csv` dans `data/`.

---

## Lancer le projet

### A. Reproduire l’expérience (entraînement + métriques)

```bash
python3 src/train_evaluate.py
```

Cela produit :
- `results/metrics_all.csv`
- les figures dans `results/figures/`
- les modèles dans `models/`
- la validation H1 / H2 / H3 dans `results/hypotheses_validation.json`

> ⏱ Durée indicative : quelques minutes (Random Forest est le plus long).



### C. Ouvrir l’application web

```bash
streamlit run app/streamlit_app.py
```

Puis ouvrir l’URL affichée (souvent `http://localhost:8501`).

---

## Exemple d’utilisation (Streamlit)

1. Lancer l’app  
2. Coller un avis, par ex. :  
   `This film was a masterpiece, brilliant acting and stunning visuals.`  
3. Cliquer sur **Analyser le sentiment**  
4. Résultat attendu : **Positif**

Essai négatif :  
`Worst movie ever, total waste of time and money.`

---

## Reproductibilité

| Paramètre | Valeur |
|-----------|--------|
| Split | 80 % train / 20 % test, **stratifié** |
| `random_state` | **42** |
| TF-IDF | `max_features=5000`, `ngram_range=(1,2)` |
| Validation | Cross-validation (5-fold ; 3-fold sous-échantillon pour RF) |

Toute personne avec le même code et le même dataset doit obtenir **les mêmes métriques**.

---

## Limites (transparence scientifique)

- Dataset **cinéma uniquement** → pas automatiquement généralisable aux avis Amazon, hôtels, etc.  
- **TF-IDF** ne comprend pas l’ironie, le sarcasme, ni bien toutes les négations.  
- Hyperparamètres **simples** (pas de grid-search exhaustif).  
- Corpus **anglais** uniquement.

---

## Perspectives

1. Tester aussi sur **Amazon Reviews** (multi-domaines)  
2. Comparer TF-IDF à des **embeddings** (Word2Vec, Transformers)  
3. Ajouter un baseline **Deep Learning** léger (DistilBERT)  
4. Explainability (**LIME / SHAP**) pour expliquer chaque prédiction  

---



## Références clés

- Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis* (dataset IMDb).  
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing*.  
- Puh, K., & Bagic Babac, M. (2023). Predicting sentiment and rating of tourist reviews.  
- Singh & Jaiswal (2023). Machine learning approaches for sentiment analysis.  
- Mao et al. (2024). Sentiment analysis — systematic review / survey.

---

## Licence & contact

Projet académique — Master BD & IA (2025–2026).  
Pour toute question pédagogique ou technique liée au cours de **Méthodologie de recherche** : s’adresser aux auteures / à l’encadrant.

---

**SmartReview Lite** — *Quand la méthodologie de recherche rencontre un vrai pipeline Machine Learning.*
