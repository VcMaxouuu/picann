# PROGRESS — alignement de `deeppic/` et `paper/` sur la méthode finale de PIC-ANN

Fichier de suivi. En cas de reprise : relire ce fichier et `git log`, puis reprendre
à la « Tâche en cours ».

## Environnement

- Python 3.11, torch 2.14 (CPU, 4 cœurs), pytest 9. Pas de GPU.
- `pip install torch` depuis PyPI (download.pytorch.org est bloqué par le proxy).
- LaTeX : `apt-get install latexmk texlive-latex-extra texlive-science ...`.
- Lancer le code : `PYTHONPATH=. python ...` depuis la racine du dépôt.
- Tests : `PYTHONPATH=. pytest tests` (rapides), `--runslow` pour les tests lents.

## Plan

0. [x] Lecture complète de `paper/main.tex`, `paper/biblio.bib`, `deeppic/**/*.py`.
1. [ ] Audit ligne à ligne du code contre le §1 (écarts listés ci-dessous).
2. [ ] Tests `tests/` qui verrouillent les propriétés (§3.2), sur le code actuel.
3. [ ] Harnais : `benchmarks/bench.py` (époques, temps, mémoire) et les harnais de
       sélection `benchmarks/balance_sel.py`, `benchmarks/nonlin_sel.py` (absents du
       dépôt et de l'historique git : à recréer). Mesure de référence *avant*
       optimisation.
4. [ ] Optimisations (§3.3), une par une, mesurées, sélections inchangées :
       syncs hôte, `ProxGenAdam` (β1 = 0 sans `exp_avg`, corrections de biais en
       scalaires, `_foreach`), `_CHECK_EVERY` selon le device, options.
5. [ ] Ensemble actif + vérification KKT (§3.4).
6. [ ] Cas limites (§3.5) : neurone à s_k = 0 dans `selected`, modèle linéaire.
7. [ ] Article : réécriture §gauge / §lambda-away-null / §optimization, annexe de
       preuves, références cassées, bibliographie, compilation sans référence
       indéfinie (hors TODO(Max)).
8. [ ] Rapport final.

## Tâche en cours

Étape 1 (audit).

## Décisions

- Branche : travail sur `claude/keen-sagan-0v37z5`, poussée aussi sur `main`
  (autorisé par l'utilisateur).
- Pertes disponibles dans le code : `SqrtMSELoss` (régression) et `BinaryLoss`
  (classification). Pas de perte de Cox dans le code actuel (elle existait dans
  l'historique git) : les tests couvrent les deux pertes présentes.
- Les harnais `balance_sel.py` / `nonlin_sel.py` n'existent pas (ni dans le dépôt, ni
  dans l'historique) : ils seront recréés, scénarios documentés dans les scripts.

## Audit (étape 1)

(en cours)
