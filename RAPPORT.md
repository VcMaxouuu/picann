# Rapport — alignement de `deeppic/` et `paper/` sur la méthode finale de PIC-ANN

Branche `claude/keen-sagan-0v37z5`, poussée aussi sur `main`. Le détail des étapes et des mesures est dans `PROGRESS.md`. Les résultats bruts sont dans `benchmarks/results/`.

## 1. Écarts entre le code et le §1, et corrections

Point de départ : commit 49f7e67. La méthode était déjà implémentée fidèlement : sensibilité, terme de multiplicateur, rétraction avec transport des moments, β1 = 0 sur W⁽¹⁾, batch complet, même X pour la calibration et l'ajustement. Écarts trouvés :

| # | Écart | Correction |
|---|---|---|
| E1 | `selected` comptait les variables qui n'entrent que par un neurone à s_k = 0 (une colonne de W⁽²⁾ nulle suffit). `count_effective_weights` aussi. | La sélection se lit maintenant sur B = s ⊙ W⁽¹⁾, sans les données. Une unité compte si un chemin de poids non nuls la relie à la sortie : c'est exactement s_k > 0 avec une couche cachée, et vrai aux annulations exactes près au-delà. Testé. |
| E2 | Une phase arrêtée par `n_epochs` rendait un itéré hors de la section (s ≈ 1,02). | Rétraction finale ; `fit` rend toujours le représentant s = 1. Testé. |
| E3 | Synchronisations hôte à chaque époque (`degenerate.any()`, `argmax.unique()`). | Supprimées : `a` est calculé par `_jacobian` sur toutes les lignes (exact, sans division ni branche), le lot fixe de p₁ lignes virtuelles est gardé, et une seule passe backward sert à la perte et au multiplicateur. |
| E4 | `ProxGenAdam` gardait `exp_avg` pour β1 = 0 et recréait les moments corrigés à chaque pas. | Plus d'`exp_avg` si β1 = 0, corrections de biais repliées en scalaires, opérations `_foreach`. Le prox est inchangé mathématiquement (testé contre la formule de référence et contre `torch.optim.Adam`). |
| E5 | Le commentaire de `_PATIENCE` annonçait 30 époques ; c'était 40 en pratique (4 contrôles sans progrès). | Commentaire corrigé, comportement inchangé. |
| E6 | Invariant 7 : ∇_{W⁽¹⁾_k·}ℓ(θ⁰)/s_k vaut **sign(a₀ₖ)**·Xᵀr̂₀, et non Xᵀr̂₀. | Le test vérifie l'identité exacte ∇ = a₀ₖ·Xᵀr̂₀. |
| E7 | Aucune perte de Cox dans le code, alors que l'article la traite. | Hors périmètre ; les tests couvrent les deux pertes présentes. |

Ajout : l'ensemble actif (§3.4). Au démarrage de chaque phase, une colonne nulle est retirée, sauf si elle échoue au test de seuillage au niveau de la phase. En fin de chemin, les colonnes retirées sont testées à λ_DB, et celles qui échouent sont réadmises avant de relancer la phase. `selector.weight` garde sa forme et son identité. `fit_phase` appelé seul garantit le test complet à son propre niveau.

## 2. Tests (`tests/`, 153 rapides + 2 lents)

Tests en float64, sur des profondeurs de 1 à 3, avec les pertes gaussienne et binaire (et le modèle linéaire là où c'est pertinent). Ils couvrent les points §3.2.1 à §3.2.8 et les invariants 1 à 8 du §2. Tous passent.

- **Sensibilité** : `a = G/r` coïncide avec `_jacobian`, qui coïncide exactement avec autograd. Les lignes où r_i = 0 (y_i = η_i aux lignes argmax ; probabilités saturées) ne faussent ni s ni i*.
- **Multiplicateur** : il coïncide avec autograd à 1e-10 près et avec des différences finies (configurations génériques). Il reste nul sur W⁽¹⁾ et sur les biais.
- **Identités d'Euler** : vérifiées pour la perte et pour ⟨W⁽ˡ⁾, ∇P⟩ = λΣ s_k‖W⁽¹⁾_k·‖₁.
- **Rétraction** : exacte (η et J inchangés, s = 1). Les gradients transportés sont égaux aux gradients recalculés, et les moments sont transportés. Les itérés ne dépendent que de l'orbite, avec ou sans moments.
- **H₀ exact** : à 1,01 λ₀, W⁽¹⁾ reste nul et ∇_φJ ≤ 1e-14 ; à 0,99 λ₀, W⁽¹⁾ devient non nul.
- **Exemple jouet** : le schéma actuel donne u → 1. Sans le terme μ∇s, on obtient u ≈ 1,57 avec Adam, et u → 1,7548 en descente de gradient simple, ce qui confirme le §1.3.
- **KKT à la convergence** : vérifié sur le modèle linéaire, sur un réseau lisse (pente 1) et sur un réseau LeakyReLU dont l'optimum est lisse.
- **Ensemble actif** : le test de violation coïncide avec un pas prox complet. Une phase élaguée est égale à la phase complète (rtol 1e-8) quand aucune colonne ne revient. Après `fit`, les colonnes retirées passent le test à λ_DB.
- **Test lent** (100 jeux sous H₀, n = 100, p = 20, `hidden_dims=(16, 8)`, α = 0,05) : **P̂(Ŝ = ∅) = 0,96** en régression et **0,89** en binaire, pour une cible de 0,95 ± 0,022. Valeurs identiques avant et après optimisation. Le binaire est à 2,7 erreurs types sous la cible : le test passe (seuil à 3 erreurs types), mais de justesse.

**Constat important pour le test KKT.** Un réseau LeakyReLU générique a souvent son optimum sur une frontière d'activation : une préactivation descend à 1e-9, et ∇_φℓ reste autour de 1e-2 quel que soit le pas. Les conditions KKT ne tiennent alors qu'au sens de Clarke. L'article le dit dans ses limites.

## 3. Benchmarks et sélections

**La dynamique est chaotique.** En float64, un écart d'arrondi entre l'ancien et le nouveau code dépasse 1e-14 à l'époque 28, puis 1e-6 à l'époque 138, puis 1e-3 à l'époque 169 (facteur ≈ 1,19 par époque). Toute optimisation qui change un arrondi change donc la trajectoire, y compris une fusion de passes backward ou une réduction `_foreach`. L'ancien code lui-même change de trajectoire à p = 5000 entre 1 et 4 threads. Une identité des sélections graine par graine n'est donc pas atteignable. L'équivalence est établie de deux façons :
1. chaque composant est testé contre une référence à 1e-12 près ;
2. les sélections sont comparées statistiquement sur de nombreuses graines.

Harnais : 1 thread par graine, graines 0 à N−1. Ancien code (8709c45) → nouveau code :

| scénario | exacts | FP moyen | phases non conv. | époques | temps (s) |
|---|---|---|---|---|---|
| linéaire n=200 p=100 s=3, (32,32), 50 graines | 48 → **50** | 0,06 → 0 | 1 → 0 | 1297 → 1229 | 2,39 → **1,97** |
| non linéaire n=500 p=100, 50 graines | 46 → 46 | 0,04 → 0,06 | 0 → 1 | 1222 → 1229 | 2,67 → **2,53** |
| linéaire p=1000, 30 graines | 22 → **25** | 3,93 → 1,57 | 9 → 5 | 2341 → 2065 | 6,16 → **4,48** |
| linéaire p=5000, 20 graines | 10 → **14** | 41,2 → 18,8 | 14 → 7 | 2502 → 2660 | 13,2 → **10,3** |

Sélections identiques graine par graine : 48/50, 46/50, 19/30 et 7/20. Les écarts vont dans les deux sens.

`bench.py` (4 threads, 3 à 5 graines, `before.json` / `after.json`) donne des gains de temps de 0,87× à 1,19×. Sur si peu de graines, l'effet est dominé par le chaos. La mémoire est inchangée : pic RSS de 676 à 732 Mo, dominé par l'import de torch.

**Options mesurées et rejetées** :
- `_CHECK_EVERY = 1` sur CPU : deux fois moins d'époques, mais p1000 tombe à 20/30 (11,9 FP) et p5000 à 9/20 (120 FP). La règle d'arrêt compare deux valeurs ponctuelles d'un objectif qui oscille : plus on la teste souvent, plus une coïncidence la déclenche tôt. On garde 10, et c'est documenté dans le code.
- Moments de φ conservés entre les phases : résultat mitigé (p5000 à 13/20 avec un cas à 249 FP, non linéaire à 43/50).
- `torch.compile` : 45 s de compilation pour gagner 8 % sur forward+backward, avec recompilation à chaque ensemble actif.
- Test KKT après chaque phase intermédiaire (ma première version) : la phase 4 était relancée 2 à 4 fois (+45 % d'époques). Remplacé par le schéma de la spécification : test au niveau de la phase suivante, puis à λ_DB.

## 4. Harnais sur au moins 50 graines

`balance_sel.py` et `nonlin_sel.py` n'existaient ni dans le dépôt ni dans l'historique git. Je les ai recréés, avec des scénarios documentés dans `benchmarks/scenarios.py` ; l'option `--scenario` permet de les lancer sur p = 1000 ou 5000. Les résultats sont dans le tableau ci-dessus.

**Faux positif persistant.** Relance des graines non linéaires avec FP, à tol/10 (1e-5) :
- graine 4 (x64) : le FP disparaît, c'était la tolérance ;
- graine 32 (**x68**) et graine 47 (x26) : les FP persistent. Ce n'est pas la tolérance. Deux graines sur 50 font 4 %, ce qui est compatible avec α = 5 %.

Le cas « x68, graine 5 » cité dans la mission ne se reproduit pas tel quel, puisque le scénario d'origine est perdu.

**Limite de la méthode, à signaler.** Quand p ≫ n (p = 1000 ou 5000, n = 200), certaines phases ne convergent pas en 1000 époques, avec des pics de perte en début de phase (Adam repart de zéro, λ ×3,16). Elles laissent alors beaucoup de faux positifs. C'était déjà le cas dans l'ancien code (41 FP en moyenne à p = 5000) ; le nouveau fait un peu mieux, sans régler le problème.

## 5. Article

Section par section :
- **Résumé** : une phrase sur la résolution dans la jauge s = 1. « Unpenalized learning phase » est remplacé par le chemin à chaud.
- **Contributions** : `lem:rankone` renvoie à la preuve du Thm `thm:ztffunc`, `prop:invariance` devient `prop:penalty-scale-invariance`, `tab:losses` devient `tab:pivotal-ztf`, `sec:optim` devient `sec:optimization`. Deux points ajoutés : les formulations équivalentes et le nouveau schéma d'optimisation.
- **Related Work, « Rescaling invariance »** : réécrit avec Path-SGD, Absil et al., ENorm, Teleportation, WeightNorm, Du–Hu–Lee, PathProx (« un PathProx dont le facteur de jauge dépend des données »), ProxGen et Combettes–Vũ.
- **§sec:gauge** : groupe 𝒢, action (éq. `group-action`), Lemme du représentant canonique.
- **§sec:learning** : (P) écrit avec λ générique et ℓ(θ). Nouvelle **§3.4 Equivalent Formulations** : (C), (Q), B, définition des points réguliers et des points du premier ordre, Proposition `prop:formulations`.
- **§sec:lambda-away-null** : le score normalisé est invariant, et sur (C) le test devient |g_kj| ≤ λ.
- **§sec:optimization** (label harmonisé) : Reversed Warm Path conservé ; Structure (Lemme d'Euler, éq. ∇_φJ, Prop. « pas de point fixe parcimonieux sans multiplicateur », exemple jouet, lecture par le multiplicateur) ; Gauge-Fixed Proximal Gradient (itération, pas effectif δs², dérive de Du et al., équivariance, transport, liens avec ENorm, Teleportation, WeightNorm et PathProx) ; Adaptive Proximal Step ; Properties (Prop. (a) à (d) et Limitations) ; Computation (a = G/r, passe linéarisée, coût, ensemble actif avec glmnet et strong rules, **Algorithme 1**) ; Optional Refitting conservé.
- **Annexe A** : preuves du Lemme d'Euler, de la Prop. sans point fixe, de la Prop. des formulations et de la Prop. des propriétés (a) à (d).
- **Bibliographie** : 11 entrées ajoutées. Champs vérifiés en ligne quand c'était possible, et DOI seulement s'ils étaient vérifiés (Davis2020, StrongRules). **Correction : Du, Hu & Lee 2018 est aux pp. 382–393**, et non 384–395. L'accent de Bằng Công Vũ a été corrigé pour que BibTeX compile.
- **Écart de placement** : la Proposition des formulations équivalentes est dans une nouvelle §3.4, juste après la définition de (P), et non dans §sec:gauge : (P) n'y est pas encore défini. Le lemme du représentant canonique, lui, est bien dans §sec:gauge.
- **Compilation** : `latexmk -pdf` (le PDF est commité). Il ne reste que les 3 références TODO(Max) indéfinies, plus un débordement de 1,4 pt dans le Thm `thm:ztffunc`, que je n'ai pas touché.

**TODO(Max)** (commentaires `% TODO(Max)`, les `??` sont visibles dans le PDF) :
1. `thm:consistency` : ce résultat n'existe pas.
2. `prop:admissible` : ce résultat n'existe pas.
3. `cor:arch` : ce résultat n'existe pas ; il découlerait du Thm `thm:ztffunc`, puisque λ₀ = ‖Xᵀr̂₀‖∞.

**Points de preuve douteux** :
- **Thm `thm:ztffunc`** (non modifié) : la preuve ne vérifie que le premier ordre. Le sens « si » demande λ > λ₀ strictement, ou un argument du second ordre à l'égalité, ainsi que des masques localement constants en θ⁰ (b⁽¹⁾_k ≠ 0, préactivations profondes non nulles).
- **§1.1/§1.2 de la mission** : « s_k homogène de degré 1 en chaque W⁽ˡ⁾ » n'est vrai qu'à masques gelés, et c'est ainsi que l'article l'énonce. L'identification (Q) ↔ Σ est ensembliste, pas topologique, car s est discontinu.
- **Propriété (c)** : elle ne couvre que les points réguliers. L'article le dit.

## 6. Changements d'API proposés, non effectués

- `fit_phase` : proposer un argument `check_kkt: bool = True` pour rendre public le comportement de `fit`, qui ne relance qu'en dernière phase. Aujourd'hui, `fit` passe par la méthode privée `_fit_phase`.
- Exposer la sensibilité finale (par exemple un buffer `unit_scale`) permettrait de lire B = s ⊙ W⁽¹⁾ avec les données. Cela changerait `state_dict`, donc je ne l'ai pas fait.
- Ajouter la perte de Cox (`Regressor.families`), traitée par l'article mais absente du code.

## 7. Simplification de l'article (demande suivante)

22 → 16 pages. On ne garde que ce qui sert : une seule formulation (le problème contraint (C), s = 1), une itération en trois gestes (normaliser, pas proximal sur W⁽¹⁾, pas Adam sur φ avec le terme μ∇s), deux garanties (points fixes = conditions du premier ordre ; point nul fixe ssi λ ≥ λ₀), et l'implémentation en trois paragraphes courts avec un algorithme réduit. Retirés : formulations (Q) et B, groupe 𝒢 et lemme du représentant canonique, points réguliers, lemme d'Euler (réduit à une identité dans l'annexe), équivariance, transport détaillé, dérive de Du et al., discussions WeightNorm/ENorm/Teleportation dans le texte principal, preuve des formulations. Le paragraphe « Rescaling invariance » du Related Work tient en un paragraphe. Version longue conservée dans l'historique git (commit f0df37c).

## 8. Travail sur (C) (demande suivante)

(P) est défini, puis (C) et leur équivalence (§3.4 « The Normalized Problem ») ; ensuite tout est énoncé sur (C) : gradient de la première couche, conditions du premier ordre de (C) avec multiplicateurs et test de seuillage (éq. 7–8), calibration, Théorème `thm:ztffunc` réénoncé sur le réseau normalisé (plus de division par s_k ; la conclusion porte sur la condition de seuillage de (C), ce que la preuve établit réellement, au lieu de « minimiseur local »), interprétation hors du nul sans ã, itération présentée comme pas + projection sur la contrainte, terme μ∇s présenté comme gradient du lagrangien de (C) avec le Lemme `lem:multipliers` (ν = μ nécessairement ; ν = 0 ⇒ W⁽¹⁾ = 0), Proposition des garanties sur (C). 16 pages.
