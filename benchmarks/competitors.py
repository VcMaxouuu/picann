"""Méthodes concurrentes pour comparer la sélection de variables.

Chaque sélecteur expose la même interface :

* ``fit(X, y)`` ;
* après ajustement, ``selected_indices`` : liste triée des indices de
  colonnes retenues ;
* ``predict(X)``, pour les métriques de prédiction éventuelles.

``task`` vaut ``"regression"`` ou ``"classification"`` (binaire, deux
classes quelconques dans ``y``).

Pour les MLP (LassoNet, STG, DeepPINK), l'entraînement suit au plus près
celui de deeppic : Adam avec ``lr = 1e-2``, batch complet, au plus 1000
époques par ajustement, pas de réajustement après sélection. LassoNet et
STG standardisent ``X`` en interne (moyenne 0, variance 1 au sens 1/n) ;
DeepPINK garde ``X`` tel quel, les knockoffs devant suivre la loi de ``X``.
En régression, les trois MLP centrent-réduisent aussi ``y`` en interne :
leur perte quadratique et leur réglage de la pénalité deviennent
indépendants de l'échelle de ``y`` ; les prédictions sont remises à
l'échelle d'origine.

Dépendances : numpy, scikit-learn, joblib, torch, lassonet, stg, Boruta,
knockpy (avec tqdm, sortedcontainers, h5py et lifelines, requis par
lassonet et stg).
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

__all__ = [
    "LassoNetSelector",
    "STGSelector",
    "BorutaSelector",
    "LassoCVSelector",
    "DeepPINKSelector",
]

TASKS = ("regression", "classification")


# ----------------------------------------------------------------------
# Outils communs
# ----------------------------------------------------------------------
def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"task doit valoir {TASKS}, reçu {task!r}")


def _splitter(task: str, n_folds: int, random_state):
    """K plis mélangés, stratifiés en classification."""
    cls = StratifiedKFold if task == "classification" else KFold
    return cls(n_splits=n_folds, shuffle=True, random_state=random_state)


def _as_arrays(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X doit être de taille (n, p) et y de taille (n,)")
    return X, y


def _encode_binary(y):
    classes, y_enc = np.unique(y, return_inverse=True)
    if classes.size != 2:
        raise ValueError(f"classification binaire : 2 classes attendues, {classes.size} trouvées")
    return classes, y_enc.astype(np.int64)


class _StandardizedMLP:
    """Standardisation interne de X (et de y en régression)."""

    task: str

    def _prepare_fit(self, X, y):
        X, y = _as_arrays(X, y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        if self.task == "classification":
            self.classes_, ys = _encode_binary(y)
        else:
            y = y.astype(np.float64)
            self.y_mean_ = float(y.mean())
            self.y_std_ = float(y.std()) or 1.0
            ys = ((y - self.y_mean_) / self.y_std_).astype(np.float32)
        return Xs, ys

    def _transform(self, X):
        return self.scaler_.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    def _to_original(self, pred):
        pred = np.asarray(pred).ravel()
        if self.task == "classification":
            return self.classes_[pred.astype(np.int64)]
        return pred * self.y_std_ + self.y_mean_


# ----------------------------------------------------------------------
# LassoNet (Lemhadri et al., JMLR 2021)
# ----------------------------------------------------------------------
def _lassonet_estimator(params):
    """LassoNetRegressor ou LassoNetClassifier (sans CV) aux réglages voulus."""
    import torch
    from lassonet import LassoNetClassifier, LassoNetRegressor

    adam = partial(torch.optim.Adam, lr=params["lr"])
    if params["path_optimizer"] == "adam":
        path_optim = adam
    elif params["path_optimizer"] == "sgd":
        path_optim = partial(torch.optim.SGD, lr=1e-3, momentum=0.9)
    else:
        raise ValueError("path_optimizer doit valoir 'adam' ou 'sgd'")
    cls = LassoNetClassifier if params["task"] == "classification" else LassoNetRegressor
    return cls(
        hidden_dims=tuple(params["hidden_dims"]),
        optim=(adam, path_optim),
        n_iters=(params["n_epochs"], 100),
        path_multiplier=params["path_multiplier"],
        batch_size=None,  # batch complet
        verbose=0,
        random_state=params["random_state"],
        torch_seed=params["random_state"],
        device=torch.device(params["device"]),
    )


def _lassonet_score(model, X, y, task):
    """Score de validation, plus grand = meilleur : R² ou −log-vraisemblance."""
    if task == "classification":
        proba = np.asarray(model.predict_proba(X))[:, 1]
        return -log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])
    return model.score(X, y)


def _lassonet_fold_path(params, X, y, train, test, n_threads):
    """Chemin LassoNet sur un pli ; renvoie les lambdas et les scores de validation."""
    import torch

    if n_threads is not None:
        torch.set_num_threads(n_threads)
    model = _lassonet_estimator(params)
    lambdas, scores = [], []

    def callback(est, hist):
        lambdas.append(hist[-1].lambda_)
        scores.append(_lassonet_score(est, X[test], y[test], params["task"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avertissements de lambda_start
        model.path(X[train], y[train], return_state_dicts=False, callback=callback)
    return lambdas, scores


class LassoNetSelector(_StandardizedMLP):
    """LassoNet avec choix de lambda par validation croisée sur le chemin.

    :param hidden_dims: largeurs des couches cachées, par exemple ``(32, 32)``.
    :param task: ``"regression"`` ou ``"classification"``.
    :param n_folds: nombre de plis de la validation croisée (5 par défaut).
    :param lr: pas d'Adam (1e-2, comme deeppic).
    :param n_epochs: époques maximales de l'entraînement dense initial.
        Chaque pas du chemin garde la valeur de LassoNet (100 époques au
        plus, arrêt précoce).
    :param path_multiplier: facteur entre deux lambdas successifs du
        chemin. 1.02 est la valeur par défaut de LassoNet ; 1.05 divise
        environ par 2,5 la longueur du chemin.
    :param path_optimizer: ``"adam"`` (Adam, ``lr``, comme deeppic) ou
        ``"sgd"`` (optimiseur natif du chemin LassoNet : SGD, lr = 1e-3,
        momentum 0.9).
    :param n_jobs: processus pour les plis (-1 : tous les cœurs).
    :param random_state: graine des plis, de la validation interne et de
        l'initialisation.
    :param device: ``"cpu"`` ou ``"cuda"`` (avec ``n_jobs=1`` sur GPU).

    La validation croisée reproduit celle de ``LassoNetCV`` (chemin par
    pli, scores interpolés sur une grille commune de lambdas, chemin final
    sur toutes les données jusqu'au meilleur lambda), avec deux
    différences : les plis tournent en parallèle, et en classification le
    score est la log-vraisemblance de validation plutôt que le taux de
    bonne classification, trop grossier (ses ex æquo font choisir le
    modèle le plus dense). En régression, le score reste le R².
    """

    def __init__(
        self,
        hidden_dims=(20,),
        task="regression",
        n_folds=5,
        lr=1e-2,
        n_epochs=1000,
        path_multiplier=1.02,
        path_optimizer="adam",
        n_jobs=-1,
        random_state=0,
        device="cpu",
    ):
        self.hidden_dims = hidden_dims
        self.task = task
        self.n_folds = n_folds
        self.lr = lr
        self.n_epochs = n_epochs
        self.path_multiplier = path_multiplier
        self.path_optimizer = path_optimizer
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        _check_task(self.task)
        Xs, ys = self._prepare_fit(X, y)
        params = dict(
            task=self.task,
            hidden_dims=tuple(self.hidden_dims),
            lr=self.lr,
            n_epochs=self.n_epochs,
            path_multiplier=self.path_multiplier,
            path_optimizer=self.path_optimizer,
            random_state=self.random_state,
            device=self.device,
        )
        folds = _splitter(self.task, self.n_folds, self.random_state).split(Xs, ys)
        n_threads = None if self.n_jobs == 1 else 1
        paths = Parallel(n_jobs=self.n_jobs)(
            delayed(_lassonet_fold_path)(params, Xs, ys, train, test, n_threads) for train, test in folds
        )

        # Grille commune et scores interpolés, comme LassoNetCV (le premier
        # point de chaque chemin est le modèle dense, lambda = 0).
        lam, lam_max = min(l[1] for l, _ in paths), max(l[-1] for l, _ in paths)
        grid = []
        while lam < lam_max:
            grid.append(lam)
            lam *= self.path_multiplier
        scores = np.stack(
            [np.interp(np.log(grid), np.log(l[1:]), s[1:]) for l, s in paths], axis=-1
        )
        best = int(np.nanargmax(scores.mean(axis=1)))
        self.lambdas_ = np.asarray(grid)
        self.cv_scores_ = scores
        self.best_lambda_ = float(grid[best])

        self.model_ = _lassonet_estimator(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = self.model_.path(Xs, ys, lambda_seq=grid[: best + 1], return_state_dicts=False)
        mask = np.asarray(path[-1].selected.cpu()) if hasattr(path[-1].selected, "cpu") else np.asarray(path[-1].selected)
        self.selected_indices = np.flatnonzero(mask).tolist()
        return self

    def predict(self, X):
        return self._to_original(self.model_.predict(self._transform(X)))


# ----------------------------------------------------------------------
# STG, stochastic gates (Yamada et al., ICML 2020)
# ----------------------------------------------------------------------
def _import_stg():
    """Importe STG. Le paquet stg 0.1.2 de PyPI utilise ``collections.Sequence``
    et consorts, retirés en Python 3.10 : on rétablit ces alias avant l'import."""
    import collections
    import collections.abc

    for name in ("Sequence", "Mapping", "Iterable", "Set"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))
    from stg import STG

    return STG


def _stg_fit(Xtr, ytr, task, hidden_dims, lam, lr, n_epochs, sigma, seed, device, n_threads):
    """Un ajustement STG complet : batch complet, Adam, sans weight decay."""
    import torch

    STG = _import_stg()

    if n_threads is not None:
        torch.set_num_threads(n_threads)
    torch.manual_seed(seed)
    classification = task == "classification"
    model = STG(
        device=device,
        input_dim=Xtr.shape[1],
        output_dim=2 if classification else 1,
        hidden_dims=list(hidden_dims),
        activation="relu",
        sigma=sigma,
        lam=float(lam),
        optimizer="Adam",
        learning_rate=lr,
        batch_size=Xtr.shape[0],
        weight_decay=0.0,  # deeppic n'a pas de weight decay
        task_type=task,
    )
    y_fit = ytr if classification else ytr.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, y_fit, nr_epochs=n_epochs, verbose=False, shuffle=False)
    return model


def _stg_output(model, X, task):
    """Sortie déterministe (portes à leur moyenne) : P(classe 1) ou prédiction."""
    import torch

    net = model._model
    net.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Softmax sans dim explicite dans stg
        out = net({"input": torch.from_numpy(X).to(model.device)})
    if task == "classification":
        return out["prob"][:, 1].cpu().numpy()
    return out["pred"].cpu().numpy().ravel()


def _stg_cv_loss(X, y, train, test, task, hidden_dims, lam, lr, n_epochs, sigma, seed, device, n_threads):
    model = _stg_fit(X[train], y[train], task, hidden_dims, lam, lr, n_epochs, sigma, seed, device, n_threads)
    pred = _stg_output(model, X[test], task)
    if task == "classification":
        return log_loss(y[test], np.clip(pred, 1e-7, 1 - 1e-7), labels=[0, 1])
    return mean_squared_error(y[test], pred)


class STGSelector(_StandardizedMLP):
    """STG avec choix de lambda par validation croisée sur une grille.

    :param hidden_dims: largeurs des couches cachées.
    :param task: ``"regression"`` ou ``"classification"``.
    :param n_folds: nombre de plis (5 par défaut).
    :param lambdas: grille de lambdas ; par défaut 10 valeurs
        log-espacées de 1e-3 à 1 (y étant centré-réduit, la perte est
        d'ordre 1 et la pénalité, une moyenne de probabilités, vaut au
        plus 1).
    :param lr: pas d'Adam (1e-2).
    :param n_epochs: époques par ajustement (1000).
    :param sigma: écart-type du bruit des portes (0.5, valeur de l'article).
    :param gate_threshold: une variable est retenue si sa porte
        déterministe, ``clip(mu + 0.5, 0, 1)``, dépasse strictement ce
        seuil. 0 retient les variables effectivement utilisées par le
        modèle à l'inférence ; 0.5 est une convention plus stricte.
    :param n_jobs: nombre de processus pour la grille lambdas × plis
        (-1 : tous les cœurs). Chaque processus utilise un seul thread.
    :param random_state: graine des plis et des initialisations.
    :param device: ``"cpu"`` ou ``"cuda"`` (avec ``n_jobs=1`` sur GPU).

    Le lambda retenu minimise la perte moyenne de validation : erreur
    quadratique en régression, log-vraisemblance négative en
    classification.
    """

    def __init__(
        self,
        hidden_dims=(20,),
        task="regression",
        n_folds=5,
        lambdas=None,
        lr=1e-2,
        n_epochs=1000,
        sigma=0.5,
        gate_threshold=0.0,
        n_jobs=-1,
        random_state=0,
        device="cpu",
    ):
        self.hidden_dims = hidden_dims
        self.task = task
        self.n_folds = n_folds
        self.lambdas = lambdas
        self.lr = lr
        self.n_epochs = n_epochs
        self.sigma = sigma
        self.gate_threshold = gate_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        _check_task(self.task)
        Xs, ys = self._prepare_fit(X, y)
        lambdas = np.logspace(-3, 0, 10) if self.lambdas is None else np.asarray(self.lambdas, dtype=float)
        folds = list(_splitter(self.task, self.n_folds, self.random_state).split(Xs, ys))
        n_threads = None if self.n_jobs == 1 else 1
        common = dict(
            task=self.task,
            hidden_dims=tuple(self.hidden_dims),
            lr=self.lr,
            n_epochs=self.n_epochs,
            sigma=self.sigma,
            seed=self.random_state,
            device=self.device,
        )

        losses = Parallel(n_jobs=self.n_jobs)(
            delayed(_stg_cv_loss)(Xs, ys, train, test, lam=lam, n_threads=n_threads, **common)
            for lam in lambdas
            for train, test in folds
        )
        losses = np.asarray(losses).reshape(len(lambdas), len(folds))
        mean = losses.mean(axis=1)
        best = int(np.argmin(mean))
        self.best_lambda_ = float(lambdas[best])
        self.cv_results_ = {
            "lambda": lambdas,
            "mean_loss": mean,
            "std_loss": losses.std(axis=1),
            "fold_loss": losses,
        }

        self.model_ = _stg_fit(Xs, ys, lam=self.best_lambda_, n_threads=None, **common)
        self.gates_ = np.asarray(self.model_.get_gates(mode="prob")).ravel()
        self.selected_indices = np.flatnonzero(self.gates_ > self.gate_threshold).tolist()
        return self

    def predict(self, X):
        out = _stg_output(self.model_, self._transform(X), self.task)
        if self.task == "classification":
            out = (out > 0.5).astype(np.int64)
        return self._to_original(out)


# ----------------------------------------------------------------------
# Forêt aléatoire + Boruta (Kursa & Rudnicki, 2010)
# ----------------------------------------------------------------------
class BorutaSelector:
    """Boruta autour d'une forêt aléatoire aux paramètres par défaut.

    :param task: ``"regression"`` ou ``"classification"``.
    :param random_state: graine de Boruta et des forêts.
    :param n_jobs: cœurs utilisés par la forêt (-1 : tous). N'influence
        pas le résultat, seulement le temps de calcul.
    :param early_stopping: arrêt anticipé de Boruta (désactivé par
        défaut, comme dans BorutaPy). L'activer accélère au prix d'un
        résultat parfois différent.

    Les variables retenues sont les variables confirmées (``support_``),
    sans les variables indécises. Boruta, lui, garde ses réglages par
    défaut (``n_estimators='auto'``, ``perc=100``, ``alpha=0.05``,
    ``max_iter=100``). ``X`` n'est pas standardisé : les forêts sont
    invariantes par transformation monotone des variables.
    """

    def __init__(self, task="regression", random_state=0, n_jobs=-1, early_stopping=False):
        self.task = task
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping = early_stopping

    def _forest(self):
        cls = RandomForestClassifier if self.task == "classification" else RandomForestRegressor
        return cls(n_jobs=self.n_jobs, random_state=self.random_state)

    def fit(self, X, y):
        from boruta import BorutaPy

        _check_task(self.task)
        X, y = _as_arrays(X, y)
        if self.task == "classification":
            self.classes_, y = _encode_binary(y)
        else:
            y = y.astype(np.float64)
        self.selector_ = BorutaPy(
            self._forest(),
            n_estimators="auto",
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            verbose=0,
        )
        self.selector_.fit(X, y)
        self.selected_indices = np.flatnonzero(self.selector_.support_).tolist()
        self._X, self._y = X, y  # pour la forêt de prédiction, ajustée à la demande
        self.forest_ = None
        return self

    def predict(self, X):
        """Forêt par défaut sur les variables retenues, ajustée au premier appel."""
        X = np.asarray(X, dtype=np.float64)
        sel = self.selected_indices
        if not sel:  # aucune variable : constante (moyenne ou classe majoritaire)
            if self.task == "classification":
                return np.full(X.shape[0], self.classes_[np.bincount(self._y).argmax()])
            return np.full(X.shape[0], self._y.mean())
        if self.forest_ is None:
            self.forest_ = self._forest().fit(self._X[:, sel], self._y)
        pred = self.forest_.predict(X[:, sel])
        return self.classes_[pred] if self.task == "classification" else pred


# ----------------------------------------------------------------------
# Lasso linéaire avec validation croisée (Tibshirani, 1996)
# ----------------------------------------------------------------------
class LassoCVSelector:
    """Lasso linéaire, niveau de pénalité choisi par validation croisée.

    :param task: ``"regression"`` (lasso, ``LassoCV``) ou
        ``"classification"`` (régression logistique pénalisée ℓ1,
        ``LogisticRegressionCV``).
    :param n_folds: nombre de plis (5 par défaut), mélangés, stratifiés
        en classification.
    :param n_lambdas: taille de la grille de pénalités (100 par défaut) :
        ``alphas`` de ``LassoCV`` en régression, ``Cs`` de
        ``LogisticRegressionCV`` en classification.
    :param n_jobs: processus pour la validation croisée (-1 : tous).
    :param random_state: graine des plis (et du solveur ``saga``).

    ``X`` est standardisé en interne, comme pour les MLP : la pénalité ℓ1
    traite alors toutes les variables sur la même échelle. En régression,
    la pénalité retenue minimise l'erreur quadratique de validation ; en
    classification, la log-vraisemblance négative de validation, comme
    pour LassoNet et STG. Les variables retenues sont celles dont le
    coefficient est non nul.
    """

    def __init__(self, task="regression", n_folds=5, n_lambdas=100, n_jobs=-1, random_state=0):
        self.task = task
        self.n_folds = n_folds
        self.n_lambdas = n_lambdas
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        _check_task(self.task)
        X, y = _as_arrays(X, y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        cv = _splitter(self.task, self.n_folds, self.random_state)
        if self.task == "classification":
            self.classes_, y = _encode_binary(y)
            self.model_ = LogisticRegressionCV(
                Cs=self.n_lambdas,
                l1_ratios=(1.0,),  # pénalité ℓ1 pure
                solver="saga",
                cv=cv,
                scoring="neg_log_loss",
                max_iter=5000,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                use_legacy_attributes=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # convergence de saga aux petits C
                self.model_.fit(Xs, y)
            coef = self.model_.coef_.ravel()
            self.best_lambda_ = float(1.0 / self.model_.C_)
        else:
            self.model_ = LassoCV(alphas=self.n_lambdas, cv=cv, n_jobs=self.n_jobs, max_iter=10000)
            self.model_.fit(Xs, y.astype(np.float64))
            coef = self.model_.coef_
            self.best_lambda_ = float(self.model_.alpha_)
        self.coef_ = coef
        self.selected_indices = np.flatnonzero(coef != 0.0).tolist()
        return self

    def predict(self, X):
        pred = self.model_.predict(self.scaler_.transform(np.asarray(X, dtype=np.float64)))
        return self.classes_[pred] if self.task == "classification" else pred


# ----------------------------------------------------------------------
# DeepPINK, knockoffs model-X (Lu et al., NeurIPS 2018)
# ----------------------------------------------------------------------
def _import_deeppink():
    """Importe knockpy et corrige trois défauts de sa version 1.3.5.

    1. ``DeepPinkModel.feature_importances`` réduit le produit des matrices
       de poids à l'intérieur de sa boucle : tout réseau à deux couches
       cachées ou plus lève ``ValueError``. On recalcule
       :math:`w = W^{(0)} W^{(1)} \\cdots W^{(L)}` (activations ignorées,
       comme Lu et al.) ; en binaire, les deux logits de sortie entrent
       par leur différence.
    2. ``train_deeppink`` passe ``y.unsqueeze(-1)`` à ``CrossEntropyLoss`` :
       une réponse binaire lève ``RuntimeError``. On entraîne le cas
       binaire avec une cible entière, par ailleurs à l'identique.
    3. ``DeepPinkStatistic.fit`` score toujours le modèle en erreur
       quadratique sur l'échantillon d'apprentissage, ce qui échoue sur
       les deux logits binaires ; ce score ne sert pas au filtre, on le
       désactive.

    Les corrections sont appliquées une seule fois par processus.
    """
    import torch
    from knockpy import knockoff_stats
    from knockpy.knockoff_filter import KnockoffFilter
    from knockpy.kpytorch import deeppink
    from torch import nn

    if getattr(deeppink, "_picann_patched", False):
        return KnockoffFilter

    def feature_importances(self, weight_scores=True):
        with torch.no_grad():
            if weight_scores:
                linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
                w = linears[0].weight.T
                for layer in linears[1:]:
                    w = w @ layer.weight.T
                w = (w[:, 0] if w.shape[1] == 1 else w[:, 1] - w[:, 0]).numpy()
            else:
                w = np.ones(self.p)
            z = self._fetch_Z_weight().numpy()
            return np.concatenate([z[self.feature_inds] * w, z[self.ko_inds] * w])

    train_gaussian = deeppink.train_deeppink

    def train_deeppink(model, features, y, **kwargs):
        if model.y_dist == "gaussian":
            return train_gaussian(model, features, y, **kwargs)
        n, p = features.shape[0], features.shape[1] // 2
        lambda1 = kwargs.get("lambda1") or 10 * np.sqrt(np.log(p) / n)
        batchsize = min(n, kwargs.get("batchsize", 100))
        features = torch.tensor(features).float()
        target = torch.tensor(y).long()
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-3))
        criterion = nn.CrossEntropyLoss(reduction="sum")
        for _ in range(kwargs.get("num_epochs", 50)):
            for Xb, yb in deeppink.create_batches(features, target, batchsize):
                loss = criterion(model(Xb), yb) + lambda1 * model.l1norm()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    deeppink.DeepPinkModel.feature_importances = feature_importances
    deeppink.train_deeppink = train_deeppink
    knockoff_stats.DeepPinkStatistic.cv_score_model = lambda self, features, y, cv_score: None
    deeppink._picann_patched = True
    return KnockoffFilter


def _fit_mlp(X, y, task, hidden_dims, lr, n_epochs, seed):
    """MLP dense (ReLU), Adam, batch complet, sans pénalité : sert à prédire
    à partir des variables retenues."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dims = [X.shape[1], *hidden_dims]
    layers = []
    for width, next_width in zip(dims, dims[1:]):
        layers += [nn.Linear(width, next_width), nn.ReLU()]
    layers.append(nn.Linear(dims[-1], 2 if task == "classification" else 1))
    net = nn.Sequential(*layers)
    Xt = torch.from_numpy(X.astype(np.float32))
    if task == "classification":
        yt, criterion = torch.from_numpy(y.astype(np.int64)), nn.CrossEntropyLoss()
    else:
        yt, criterion = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1), nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        criterion(net(Xt), yt).backward()
        optimizer.step()
    return net.eval()


class DeepPINKSelector:
    """DeepPINK : knockoffs model-X gaussiens et MLP à couche de couplage,
    filtre knockoff+ au taux de fausses découvertes ``fdr``.

    :param hidden_dims: largeurs des couches cachées du MLP placé après la
        couche de couplage par paires.
    :param task: ``"regression"`` ou ``"classification"``.
    :param fdr: taux de fausses découvertes visé ``q`` (0.2 par défaut).
    :param Sigma: covariance des lignes de ``X`` ; ``None`` l'estime
        (Ledoit-Wolf, défaut de knockpy), comme sur données réelles. En
        simulation, passer la vraie covariance.
    :param lr: pas d'Adam (1e-2, comme deeppic).
    :param n_epochs: époques d'entraînement (1000), en batch complet.
    :param normalize_Z: normalisation propre à knockpy des poids de
        couplage de chaque paire variable/knockoff ; ``False`` (défaut)
        garde les poids libres de Lu et al.
    :param random_state: graine des knockoffs (NumPy) et du réseau
        (PyTorch).

    L'entraînement est celui de knockpy : perte quadratique (régression)
    ou entropie croisée (classification), sommée sur l'échantillon, plus
    une pénalité ℓ1 sur tous les poids de niveau
    :math:`10 \\sqrt{\\log p / n}`. La statistique d'importance est celle
    de Lu et al., :math:`Z_j = z_j w_j`, et la statistique de knockoff
    :math:`|Z_j| - |\\tilde Z_j|`.

    Le filtre knockoff+ ne retient rien tant qu'il ne peut pas faire au
    moins ``1 / fdr`` découvertes, soit 5 avec ``q = 0.2`` : avec moins de
    cinq variables pertinentes, DeepPINK ne sélectionne rien par
    construction. ``X`` n'est pas standardisé : les knockoffs doivent
    suivre la loi de ``X``. En régression, ``y`` est centré-réduit, comme
    pour LassoNet et STG : la perte étant une somme et le niveau de la
    pénalité ℓ1 fixe, l'échelle de ``y`` réglerait sinon le poids relatif
    des deux.
    ``predict`` ajuste à la demande un MLP dense de même architecture sur
    les variables retenues : le réseau de DeepPINK prend en entrée les
    knockoffs, qui n'existent pas pour de nouvelles observations.
    """

    def __init__(
        self,
        hidden_dims=(20,),
        task="regression",
        fdr=0.2,
        Sigma=None,
        lr=1e-2,
        n_epochs=1000,
        normalize_Z=False,
        random_state=0,
    ):
        self.hidden_dims = hidden_dims
        self.task = task
        self.fdr = fdr
        self.Sigma = Sigma
        self.lr = lr
        self.n_epochs = n_epochs
        self.normalize_Z = normalize_Z
        self.random_state = random_state

    def fit(self, X, y):
        import torch

        KnockoffFilter = _import_deeppink()
        _check_task(self.task)
        X, y = _as_arrays(X, y)
        if self.task == "classification":
            self.classes_, ys = _encode_binary(y)
        else:
            y = y.astype(np.float64)
            self.y_mean_ = float(y.mean())
            self.y_std_ = float(y.std()) or 1.0
            ys = (y - self.y_mean_) / self.y_std_

        np.random.seed(self.random_state)  # knockpy tire les knockoffs avec NumPy
        torch.manual_seed(self.random_state)
        self.filter_ = KnockoffFilter(ksampler="gaussian", fstat="deeppink")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected = self.filter_.forward(
                X=X,
                y=ys,
                Sigma=self.Sigma,
                fdr=self.fdr,
                fstat_kwargs={
                    "hidden_sizes": list(self.hidden_dims),
                    "y_dist": "binomial" if self.task == "classification" else "gaussian",
                    "normalize_Z": self.normalize_Z,
                    "train_kwargs": {
                        "num_epochs": self.n_epochs,
                        "batchsize": X.shape[0],  # batch complet
                        "lr": self.lr,
                        "verbose": False,
                    },
                },
            )
        self.W_ = np.asarray(self.filter_.W)
        self.threshold_ = float(self.filter_.threshold)
        self.selected_indices = np.flatnonzero(np.asarray(selected, dtype=bool)).tolist()
        self._X, self._y = X, ys
        self.net_ = None
        return self

    def predict(self, X):
        """MLP dense sur les variables retenues, ajusté au premier appel."""
        import torch

        X = np.asarray(X, dtype=np.float64)
        sel = self.selected_indices
        if not sel:  # aucune variable : constante (moyenne ou classe majoritaire)
            if self.task == "classification":
                return np.full(X.shape[0], self.classes_[np.bincount(self._y).argmax()])
            return np.full(X.shape[0], self.y_mean_)
        if self.net_ is None:
            self.scaler_ = StandardScaler().fit(self._X[:, sel])
            self.net_ = _fit_mlp(
                self.scaler_.transform(self._X[:, sel]),
                self._y,
                self.task,
                tuple(self.hidden_dims),
                self.lr,
                self.n_epochs,
                self.random_state,
            )
        with torch.no_grad():
            out = self.net_(torch.from_numpy(self.scaler_.transform(X[:, sel]).astype(np.float32)))
        if self.task == "classification":
            return self.classes_[out.argmax(dim=1).numpy()]
        return out.numpy().ravel() * self.y_std_ + self.y_mean_
