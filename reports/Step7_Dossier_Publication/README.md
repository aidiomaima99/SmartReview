# Step 7 — Dossier « Prêt à publier »

Livrable de l'étape *Publication et Dissémination des résultats* (SmartReview, Master BD & IA,
Méthodologie de la Recherche, Prof. M. EL HAJJI). Contenu généré à partir des résultats réels
produits aux Steps 5–6 (`results/metrics_all.csv`, `results/hypotheses_validation.json`,
`results/figures/*.png`) — aucune donnée n'a été inventée.

## Contenu du dossier

| Fichier | Rôle | Correspond à |
|---|---|---|
| `Step7_1_Manuscript_IEEE.pdf` | Article scientifique complet, mise en page 2 colonnes façon IEEE (titre, auteurs, abstract, 8 sections, 4 figures, 4 tableaux, 6 références), **4 pages** | Activité A |
| `Step7_1_Manuscript_IEEE.tex` | Source LaTeX du même article sur le template standard **IEEEtran** (classe `IEEEtran`, `conference`) — compilable sur Overleaf ou avec `pdflatex` | Activité A (exigence « template standard ») |
| `Step7_2_Fiche_Evaluation_Canal.pdf` | Audit du canal de publication cible (**IEEE ICMLA**, conférence — avec **IEEE Access** en piste revue à moyen terme) : identification, indexations réelles (IEEE Xplore, Scopus, DBLP, CORE C), vérification anti-prédateur sur la Beall's List (beallslist.net) datée du 17/07/2026, grille Think.Check.Submit, justification en 2 lignes | Activité B |
| `Step7_3_Slides_Presentation.pdf` | Support de soutenance flash, 5 slides (titre, problématique/hypothèses, méthodologie, résultats, conclusion/perspectives) | Activité D |

## Comment le dossier a été produit

Les trois PDF sont générés par un script Python unique. La Fiche et les Slides reprennent le
système visuel du support de soutenance de référence du projet (bandes dégradées bordeaux/rose,
cartes arrondies, badges de statut H1/H2/H3, pilules "STEP"). Le Manuscript reste sobre et
noir-et-blanc, conformément aux standards IEEE, avec un liseré bordeaux dans les tableaux pour
rester cohérent avec le reste du dossier :

```bash
python src/generate_step7_dossier.py
```

Source : [`src/generate_step7_dossier.py`](../../src/generate_step7_dossier.py). Le script lit
directement `results/metrics_all.csv` et `results/hypotheses_validation.json`, donc si
l'expérimentation (`src/train_evaluate.py`) est relancée, il suffit de relancer
`generate_step7_dossier.py` pour que le manuscrit, la fiche et les slides se mettent à jour avec
les nouveaux chiffres.

## Résultat clé rappelé

Sous la condition « avec prétraitement NLP » : **Logistic Regression** obtient le meilleur
F1-score (0.8917, Accuracy 89.03 %), en 1.72 s. H1 est réfutée (SVM ne domine pas), H2 n'est pas
confirmée (le prétraitement n'améliore aucun des 4 modèles), H3 est confirmée (LR = meilleur
compromis performance/temps). Détails complets : voir `reports/Steps/Step6_Analyse_Resultats.pdf`
et le manuscrit IEEE de ce dossier.

## Canal de publication retenu

**IEEE ICMLA** (International Conference on Machine Learning and Applications — conférence IEEE,
actes ISSN 1946-0740) — indexée IEEE Xplore, Scopus, DBLP, classée CORE C ; absente de la Beall's
List consultée le 17/07/2026. **IEEE Access** (revue, ISSN 2169-3536, Scopus/WoS/DOAJ) est
conservée comme piste réaliste à moyen terme pour une version étendue de l'article. Détails et
preuves dans `Step7_2_Fiche_Evaluation_Canal.pdf`.

## Note sur les chiffres utilisés

Deux jeux de résultats existent dans l'historique du dépôt pour la Régression Logistique
(condition « avec prétraitement ») : un run antérieur (F1=0.889, Acc=88.71 %, repris dans le
README racine et dans certains supports externes) et le run le plus récent, dont les horodatages
sont postérieurs de ~1h (`results/metrics_all.csv`, `results/hypotheses_validation.json`,
2026‑07‑10 13:23) et qui est celui reflété dans `reports/Steps/Step6_Analyse_Resultats.pdf`
(F1=0.8917, Acc=89.03 %, 1.72 s). Ce dossier Step 7 utilise ce second jeu, le plus récent et le
seul aligné avec le Step 6 déjà validé.
