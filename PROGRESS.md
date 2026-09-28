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

Étape 3 (harnais de mesure, référence avant optimisation).

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
