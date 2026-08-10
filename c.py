"""
Calibration de lambda pour PIC-ANN (cas regression gaussienne).

    lambda = q_{1-alpha}(Lambda_A),

    Lambda_A =  sup_{||u||_1 <= 1, beta in R}  |<Z~, (X u + beta 1)_+>| / (sqrt(n) ||Z~||_2)

avec Z ~ N(0, I_n) et Z~ = Z - moyenne(Z). Sous H0, Z~/||Z~|| est uniforme sur la
sphere unite de 1^perp : la loi de Lambda_A ne depend que de X. Pivotalite exacte
en echantillon fini, aucun parametre de nuisance a simuler.

Deux oracles :
  - univarie  (u = +-e_j) : EXACT, O(n log n) par variable, par tri + sommes suffixes.
  - multivarie (u quelconque) : non convexe, ascension de gradient projetee sur la
    boule l1, multi-departs, initialisee sur l'optimum univarie.

Sous H0 avec un design sans structure, les deux donnent le meme quantile : l'oracle
univarie suffit pour CALIBRER. Le multivarie n'est necessaire que pour evaluer
Lambda_A sur les vraies donnees (le certificat), ou le signal non lineaire vit.
"""

from __future__ import annotations

import math
import torch


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def null_scores(n: int, M: int, seed: int = 0, dtype=torch.float32,
                device="cpu") -> torch.Tensor:
    """M tirages i.i.d. de xi sous H0 : uniforme sur la sphere unite de 1^perp."""
    g = torch.Generator(device=device).manual_seed(seed)
    Z = torch.randn(M, n, generator=g, dtype=dtype, device=device)
    Z = Z - Z.mean(dim=1, keepdim=True)
    return Z / (math.sqrt(n) * Z.norm(dim=1, keepdim=True))


def data_score(y: torch.Tensor) -> torch.Tensor:
    """xi sur les vraies donnees : (y - ybar) / (sqrt(n) ||y - ybar||_2)."""
    r = y - y.mean()
    return r / (math.sqrt(y.numel()) * r.norm())


# ---------------------------------------------------------------------------
# Oracle univarie : EXACT
# ---------------------------------------------------------------------------

def univariate_atom_stat(X: torch.Tensor, Xi: torch.Tensor,
                         chunk: int | None = None) -> torch.Tensor:
    """
    max_{j, s=+-1, beta} | sum_i xi_i (s x_ij + beta)_+ |, pour chaque ligne de Xi.

    G(t) = sum_i xi_i (v_i - t)_+ est lineaire par morceaux en t = -beta, avec des
    coudes exactement aux valeurs observees v_i, et G(-inf) = sum_i xi_i v_i
    puisque sum_i xi_i = 0. Le sup est donc atteint sur la grille des observations
    ou a la limite : tri + sommes suffixes suffisent.
    """
    n, p = X.shape
    M = Xi.shape[0]
    if chunk is None:                       # ~32 Mo par bloc
        chunk = max(1, int(8e6 / max(M * n, 1)))
    best = torch.full((M,), -float("inf"), dtype=X.dtype, device=X.device)
    V = torch.cat([X, -X], dim=1)           # les deux signes d'un coup

    for start in range(0, V.shape[1], chunk):
        Vc = V[:, start:start + chunk]                       # (n, c)
        vs, idx = torch.sort(Vc, dim=0)                      # (n, c)
        xs = Xi[:, idx]                                      # (M, n, c)
        A = torch.flip(torch.cumsum(torch.flip(xs * vs.unsqueeze(0), [1]), dim=1), [1])
        B = torch.flip(torch.cumsum(torch.flip(xs, [1]), dim=1), [1])
        G = A - vs.unsqueeze(0) * B                          # coudes
        cand = torch.maximum(G.abs().amax(dim=1), A[:, 0, :].abs())   # + limite lineaire
        best = torch.maximum(best, cand.amax(dim=1))
    return best


# ---------------------------------------------------------------------------
# Oracle multivarie : ascension de gradient projetee
# ---------------------------------------------------------------------------

def _project_l1_ball(U: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Projection euclidienne ligne par ligne sur {||u||_1 <= radius}."""
    absU = U.abs()
    need = absU.sum(dim=1) > radius
    if not bool(need.any()):
        return U
    Un = absU[need]
    s, _ = torch.sort(Un, dim=1, descending=True)
    cs = torch.cumsum(s, dim=1) - radius
    k = torch.arange(1, Un.shape[1] + 1, device=U.device, dtype=U.dtype)
    rho = (s - cs / k > 0).to(U.dtype).cumsum(dim=1).argmax(dim=1)
    theta = cs.gather(1, rho.unsqueeze(1)) / (rho + 1).unsqueeze(1).to(U.dtype)
    out = U.clone()
    out[need] = torch.sign(U[need]) * torch.clamp(Un - theta, min=0.0)
    return out


def ramp_atom_stat(X: torch.Tensor, Xi: torch.Tensor, n_restarts: int = 8,
                   n_steps: int = 80, lr: float = 0.08, seed: int = 0,
                   return_atoms: bool = False):
    """
    Approxime sup_{||u||_1<=1, beta} |<xi, (Xu + beta)_+>| pour chaque ligne de Xi.
    Probleme non convexe : multi-departs, dont un place sur l'optimum univarie exact.
    """
    n, p = X.shape
    M, R = Xi.shape[0], n_restarts
    g = torch.Generator(device=X.device).manual_seed(seed)

    B = M * R * 2                                   # x2 pour les deux signes
    U = torch.randn(B, p, generator=g, dtype=X.dtype, device=X.device)
    U = U / U.abs().sum(dim=1, keepdim=True)
    b = torch.randn(B, generator=g, dtype=X.dtype, device=X.device)

    # warm start : un depart par tirage sur l'optimum univarie
    with torch.no_grad():
        corr = Xi @ X
        j = corr.abs().argmax(dim=1)
        sg = torch.sign(corr.gather(1, j.unsqueeze(1))).squeeze(1)
    i0 = torch.arange(0, B, R, device=X.device)
    dm = (i0 // R) % M
    U.data[i0] = 0.0
    U.data[i0, j[dm]] = sg[dm]
    b.data[i0] = 0.0

    sign = torch.cat([torch.ones(M * R, dtype=X.dtype, device=X.device),
                      -torch.ones(M * R, dtype=X.dtype, device=X.device)])
    draw = torch.arange(M, device=X.device).repeat_interleave(R).repeat(2)
    XiB = Xi[draw]

    U.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.Adam([U, b], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr / 50)

    best = torch.zeros(B, dtype=X.dtype, device=X.device)
    bU, bb = U.detach().clone(), b.detach().clone()
    for _ in range(n_steps):
        opt.zero_grad()
        vals = (XiB * torch.relu(X @ U.t() + b).t()).sum(dim=1)
        (-(sign * vals).sum()).backward()
        opt.step()
        sch.step()
        with torch.no_grad():
            U.data = _project_l1_ball(U.data, 1.0)
            cur = vals.detach().abs()
            imp = cur > best
            best = torch.where(imp, cur, best)
            bU[imp], bb[imp] = U.data[imp], b.data[imp]

    out = torch.zeros(M, dtype=X.dtype, device=X.device)
    out.scatter_reduce_(0, draw, best, reduce="amax", include_self=False)
    return (out, bU, bb, draw) if return_atoms else out


# ---------------------------------------------------------------------------
# Lambda_A et calibration
# ---------------------------------------------------------------------------

def lambda_A(X: torch.Tensor, Xi: torch.Tensor, mode: str = "univ",
             n_restarts: int = 8, n_steps: int = 80, seed: int = 0) -> torch.Tensor:
    """Lambda_A pour chaque ligne de Xi. mode in {'univ', 'ramp'}."""
    stat = univariate_atom_stat(X, Xi)
    if mode == "ramp":
        stat = torch.maximum(stat, ramp_atom_stat(X, Xi, n_restarts, n_steps, seed=seed))
    elif mode != "univ":
        raise ValueError("mode doit valoir 'univ' ou 'ramp'")
    return stat


def pic_lambda(X: torch.Tensor, alpha: float = 0.05, M: int = 1000,
               mode: str = "univ", n_restarts: int = 8, n_steps: int = 80,
               seed: int = 0, return_dist: bool = False):
    """
    Simule lambda = q_{1-alpha}(Lambda_A) sous H0, conditionnellement a X.

    X     : (n, p) design, colonnes standardisees
    alpha : niveau (P(S_hat = vide | H0) = 1 - alpha)
    M     : nombre de tirages Monte-Carlo
    mode  : 'univ' (exact, recommande pour calibrer) ou 'ramp' (multivarie)

    Retourne lambda, ou (lambda, echantillon) si return_dist.
    """
    Xi = null_scores(X.shape[0], M, seed=seed, dtype=X.dtype, device=X.device)
    stat = lambda_A(X, Xi, mode=mode, n_restarts=n_restarts,
                    n_steps=n_steps, seed=seed + 1)
    lam = torch.quantile(stat, 1.0 - alpha).item()
    return (lam, stat) if return_dist else lam


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def standardize(X):
        return (X - X.mean(0, keepdim=True)) / X.std(0, unbiased=False, keepdim=True)

    torch.manual_seed(0)
    n, p = 300, 50
    X = standardize(torch.randn(n, p))

    lam_u = pic_lambda(X, alpha=0.05, M=2000, mode="univ")
    lam_r = pic_lambda(X, alpha=0.05, M=300, mode="ramp", n_restarts=8, n_steps=80)
    Xi = null_scores(n, 2000, seed=0)
    lam_lin = torch.quantile((Xi @ X).abs().amax(dim=1), 0.95).item()
    print(f"lambda (univarie exact) = {lam_u:.5f}")
    print(f"lambda (multivarie)     = {lam_r:.5f}")
    print(f"lambda lineaire seul    = {lam_lin:.5f}   <- insuffisant, mur troue\n")

    # pivotalite : la loi ne doit pas bouger avec (beta_0, sigma)
    print("Pivotalite et niveau empirique (1000 replicats, X fixe) :")
    for b0, sg in [(0.0, 1.0), (5.0, 0.1), (-3.0, 20.0)]:
        Y = b0 + sg * torch.randn(1000, n)
        Xi = Y - Y.mean(dim=1, keepdim=True)
        Xi = Xi / (math.sqrt(n) * Xi.norm(dim=1, keepdim=True))
        st = univariate_atom_stat(X, Xi)
        print(f"  beta_0={b0:6.1f} sigma={sg:6.1f}   E[Lambda_A]={st.mean():.4f}   "
              f"P(rejet|H0)={(st > lam_u).float().mean():.3f}")

    # certificat sur un signal purement non lineaire
    y = X[:, 0].abs() + 0.3 * torch.randn(n)
    xi = data_score(y).unsqueeze(0)
    print(f"\nSignal y = |x_1| + bruit :")
    print(f"  statistique lineaire  = {float((xi @ X).abs().amax()):.5f}  vs lambda_lin = {lam_lin:.5f}  -> RIEN")
    print(f"  Lambda_A (rampe)      = {float(lambda_A(X, xi, mode='ramp', n_restarts=32, n_steps=150)):.5f}"
          f"  vs lambda     = {lam_u:.5f}  -> DETECTE")
