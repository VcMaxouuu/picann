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
1. [x] Audit ligne à ligne du code contre le §1 (écarts listés ci-dessous).
2. [x] Tests `tests/` qui verrouillent les propriétés (§3.2), sur le code actuel.
       130 tests rapides verts ; échecs initiaux = exactement E1 et E2, corrigés
       (commit 8709c45). Test lent H0 : `--runslow`.
3. [x] Harnais recréés : `benchmarks/bench.py` + `compare.py` (époques, temps, mémoire,
       sélections ; 4 threads) et `balance_sel.py` / `nonlin_sel.py` (+ `--scenario`,
       1 thread par graine, 4 processus). Référence *avant* : worktree du commit 8709c45.
4. [x] Optimisations (§3.3) : syncs supprimées (a par `_jacobian` sur toutes les
       lignes, lot fixe de p₁ lignes virtuelles, une seule passe backward),
       `ProxGenAdam` (pas d'`exp_avg` si β1 = 0, corrections de biais en scalaires,
       `_foreach`). Options mesurées et rejetées : `_CHECK_EVERY = 1` sur CPU (E3),
       moments de φ conservés entre phases (E2), `torch.compile`.
5. [x] Ensemble actif + test KKT (§3.4) : pré-test au niveau de chaque phase, test
       final à λ_DB avec relance (commit dfbb1c0).
6. [x] Cas limites (§3.5) : neurone à s_k = 0 (E1), modèle linéaire (tests).
7. [x] Article réécrit, compilé (seules références indéfinies : les 3 TODO(Max)).
8. [x] Rapport final : `RAPPORT.md`.

## Tâche en cours

Aucune. Dernière demande (simplification du code) : faite, voir ci-dessous.

## Simplification du code (demande utilisateur, après la mission)

- Ensemble actif et tests KKT supprimés : chaque phase travaille sur tout W⁽¹⁾,
  rien n'est élagué entre les phases. `_violations`, `_run_phase`, `_descend`,
  `_fit_phase`, `_MAX_KKT_ROUNDS`, `_check_every` supprimés ; `tests/test_active_set.py`
  supprimé.
- `base.py` regroupé (977 → 699 lignes) : `_forward` (prédicteur + pentes),
  `_jacobian`, `_multiplier`, `_gradients`, `_normalize`, `_optimizer`, et la boucle
  dans `fit_phase`. Itération : gradients au point courant, normalisation (gradients
  et moments transportés : exactement les gradients au réseau normalisé, J étant
  invariant), pas lasso proximal sur W⁽¹⁾, pas Adam sur φ.
- Objectif enregistré : J_λ au point courant ; nouveau test : identique pour une
  copie rescalée. Sur un exemple non linéaire, s_k saute parfois (jusqu'au facteur
  1/m = 100) quand une activation bascule, mais sans saut visible de J (neurones
  concernés à W⁽¹⁾ ≈ 0) ; les variations de J (~0,7 %/époque) sont l'oscillation d'Adam.

Harnais (1 thread) — original (49f7e67) / ensemble actif (dfbb1c0) / simplifié :

| scénario | exact | temps (s) |
|---|---|---|
| linear_p100 (50) | 48 / 50 / 50 | 2,39 / 1,97 / 2,33 |
| nonlinear_p100 (50) | 46 / 46 / 46 | 2,67 / 2,53 / 2,61 |
| linear_p1000 (30) | 22 / 25 / 23 | 6,16 / 4,48 / 5,13 |
| linear_p5000 (20) | 10 / 14 / 9 | 13,2 / 10,3 / 13,05 |

FP moyens simplifié : 0 ; 0,06 ; 1,70 ; 18,55. Phases non convergées : 0 ; 0 ; 7 ; 12.

## Décisions

- Branche : travail sur `claude/keen-sagan-0v37z5`, poussée aussi sur `main`
  (autorisé par l'utilisateur).
- Pertes disponibles dans le code : `SqrtMSELoss` (régression) et `BinaryLoss`
  (classification). Pas de perte de Cox dans le code actuel (elle existait dans
  l'historique git) : les tests couvrent les deux pertes présentes.
- Les harnais `balance_sel.py` / `nonlin_sel.py` n'existent pas (ni dans le dépôt, ni
  dans l'historique) : ils seront recréés, scénarios documentés dans les scripts.
- E1 : `selected` lit la sélection sur B = s ⊙ W⁽¹⁾ sans les données : une unité compte
  si un chemin de poids non nuls la relie à la sortie (= s_k > 0 exactement avec une
  couche cachée, aux annulations exactes près au-delà). `count_effective_weights`
  suit la même règle.
- E2 : une phase arrêtée par `n_epochs` rétracte son dernier itéré (sans le compter
  dans l'historique) : `fit` rend toujours le représentant s = 1.
- Test KKT : un réseau LeakyReLU générique a souvent son optimum sur une frontière
  d'activation (préactivation → 1e-9 et ∇_φℓ ≈ 1e-2 qui ne diminue pas avec le pas) :
  KKT au sens de Clarke seulement. Le test utilise un problème dont l'optimum est
  lisse (n=60, p=5, graine 1) + un cas pente 1 (réseau lisse) + le modèle linéaire.
- Exemple jouet : avec le schéma actuel u → 1,00004 ; sans le terme μ∇s (ancien
  schéma), u ≈ 1,57 après 20000 époques avec Adam, et u → 1,7548 (racine de
  (u−2)(1+u²) = −1) en descente de gradient simple : l'affirmation du §1.3 est vérifiée.

## Mesures (étapes 3–5)

Chaos numérique : en float64, l'écart de trajectoire entre l'ancien et le nouveau code
(même init, phase 1) passe 1e-14 à l'époque 28, 1e-10 à 81, 1e-6 à 138, 1e-3 à 169
(≈ ×1,19 par époque). Toute modification de l'arrondi change donc les trajectoires ;
l'ancien code lui-même est déterministe à p ≤ 1000 mais pas à p = 5000 (4 vs 1
threads). La comparaison des sélections est donc statistique (harnais, 1 thread).

Harnais (graines 0..N−1) — ancien (8709c45) → nouveau (E1 = HEAD) :

| scénario | exact | FP moyen | phases non conv. | époques | temps (s) |
|---|---|---|---|---|---|
| linear_p100 (50) | 48 → 50 | 0,06 → 0 | 1 → 0 | 1297 → 1229 | 2,39 → 1,97 |
| nonlinear_p100 (50) | 46 → 46 | 0,04 → 0,06 | 0 → 1 | 1222 → 1229 | 2,67 → 2,53 |
| linear_p1000 (30) | 22 → 25 | 3,93 → 1,57 | 9 → 5 | 2341 → 2065 | 6,16 → 4,48 |
| linear_p5000 (20) | 10 → 14 | 41,2 → 18,8 | 14 → 7 | 2502 → 2660 | 13,2 → 10,3 |

Sélections identiques graine par graine : 48/50, 46/50, 19/30, 7/20 (écarts dans les
deux sens). Variantes rejetées :
- post-test KKT à *chaque* phase (première version) : phase 4 relancée 2 à 4 fois à
  p = 5000 (3865 époques, 15,8 s) ;
- E2 (moments de φ gardés entre phases) : p1000 26/30 mais p5000 13/20 avec un cas à
  249 FP, non linéaire 43/50 ;
- E3 (`_CHECK_EVERY = 1` sur CPU) : −47 % d'époques mais p1000 20/30 (FP 11,9), p5000
  9/20 (FP 120) : la règle d'arrêt compare deux valeurs d'un objectif qui oscille ;
- `torch.compile` : 45 s de compilation pour −8 % sur forward+backward, recompilation
  à chaque ensemble actif.

Test H0 lent (100 jeux, n=100, p=20, (16, 8), α = 0,05), ancien code :
P̂(Ŝ = ∅) = 0,96 (gaussien), 0,89 (binaire) ; cible 0,95 ± 0,022. Nouveau code : identique.

tol/10 sur les FP non linéaires : graine 4 corrigée ; graines 32 (x68) et 47 persistent.

## Audit (étape 1) — code d'origine (commit 49f7e67) contre le §1

### Conforme

- A1. `_backward` : `a = G / r` (G = ∂ℓ/∂h⁽¹⁾, r = ∂ℓ/∂η), lignes |r_i| ≤ eps·max|r|
  recalculées par `_jacobian` ; `s_k = max_i |a_ik|`, `i*_k = argmax`. Écart mesuré
  G/r vs `_jacobian` ≤ 1e-14 en relatif (float64) ; `_jacobian` = autograd exactement.
- A2. φ reçoit ∇_φ ℓ + Σ_k μ_k ∇_φ s_k (passe linéarisée, masques gelés, biais retirés,
  μ_k = λ‖W⁽¹⁾_k·‖₁) ; W⁽¹⁾ ne reçoit que ∇ℓ ; aucun biais ne reçoit de terme de s.
- A3. `_retract` : c_k = s_k ; W⁽¹⁾_k·, b⁽¹⁾_k × c ; W⁽²⁾_·k / c ; gradients / facteur ;
  `exp_avg` / facteur, `exp_avg_sq` / facteur² ; s_k = 0 laissé tel quel.
- A4. Groupe W⁽¹⁾ : `betas=(0, 0.999)`, pas de weight decay / Nesterov / clipping ;
  prox au seuil D·λ·s (s = 1 après rétraction).
- A5. Groupe φ : Adam (0.9, 0.999), sans weight decay ni régularisation.
- A6. Batch complet ; même X standardisée (même dtype) pour `calibrate` et les phases ;
  pas d'AMP.
- A7. Arrêt |J(t−50) − J(t)| / |J(t)| ≤ tol (vérifié toutes les 10 époques) ; chemin
  géométrique croissant avec démarrage à chaud.

### Écarts

- E1. (§3.5) `selected` compte les variables qui n'entrent que par un neurone à
  s_k = 0 (vérifié : colonne de W⁽²⁾ nulle ⇒ variable déclarée sélectionnée).
  `count_effective_weights` compte aussi ces poids.
- E2. Une phase arrêtée par `n_epochs` sans converger rend l'itéré *après* le pas
  prox-Adam, donc hors de la section (s ≈ 1,01–1,02 mesuré). `selected` n'est pas
  affecté (les zéros sont invariants), mais « après `fit`, s_k = 1 » et B = W⁽¹⁾ sont faux.
- E3. (§3.3) Synchronisations hôte à chaque époque : `degenerate.any()` et
  `argmax.unique()`.
- E4. (§3.3) `ProxGenAdam` alloue et met à jour `exp_avg` pour β1 = 0 (où il vaut
  `grad`), et crée `m/(1−β1^t)`, `v/(1−β2^t)` à chaque pas.
- E5. Commentaire de `_PATIENCE` : `ReduceLROnPlateau(patience=3)` avec un contrôle
  toutes les 10 époques divise le pas après 4 contrôles sans progrès (40 époques),
  pas 30.
- E6. Invariant 7 (§2) : ∇_{W⁽¹⁾_k·}ℓ(θ⁰)/s_k(θ⁰) = sign(a_0k)·Xᵀr̂₀, et non Xᵀr̂₀ ;
  l'identité exacte est ∇_{W⁽¹⁾_k·}ℓ(θ⁰) = a_0k·Xᵀr̂₀ (c'est ce que le test vérifie).
- E7. Pas de perte de Cox dans le code (l'article la couvre) : hors périmètre, signalé.

### Remarques mathématiques sur le §1 (à refléter dans l'article, pas des erreurs de code)

- M1. « s_k positivement homogène de degré 1 en chaque W⁽ˡ⁾ » : vrai à masques
  gelés (localement, p.p.), pas globalement (multiplier W⁽ˡ⁾ sans son biais déplace les
  masques). L'identité d'Euler pour s tient p.p.
- M2. (Q) : {s = 1} rencontre chaque orbite exactement une fois (bijection avec
  Θ₊/G), mais θ ↦ G_{s(θ)}θ est discontinue là où s saute : identification ensembliste.
- M3. Preuve du Théorème `thm:ztffunc` : seules les conditions du premier ordre sont
  vérifiées. Le sens « si » demande λ > λ₀ strictement (ou un argument du second ordre
  à l'égalité) et des masques localement constants en θ⁰ (b⁽¹⁾_k ≠ 0, préactivations
  profondes ≠ 0). Théorème non modifié (hors périmètre), signalé.
