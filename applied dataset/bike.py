# ─────────────────────────────────────────────────────────────────────────────
# EACS vs. baselines and Anchor/Oracle/ICP/Lasso on Bike Sharing
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import linalg as la
from scipy.stats import f as fdist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─── Reproducibility / Seeding ───────────────────────────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Settings ────────────────────────────────────────────────────────────────
alpha      = 0.05
n_folds    = 5
INNER_CV   = 3   # inner CV folds
gamma_grid = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
feats      = ["temp","atemp","hum","windspeed"]

PICKLE_PATH = 'bike_res.pkl'

# ─── DeepSet embedder ───────────────────────────────────────────────────────
class EnvDeepSetEmbed(nn.Module):
    def __init__(self, in_dim, phi_hidden=64, rho_hidden=64, embed_dim=32):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, embed_dim),
            nn.ReLU(),
        )

    def forward(self, X, mask=None):
        """
        X:   (B, N, p)
        mask:(B, N, 1) with 1 for real samples and 0 for padded rows.
        """
        h = self.phi(X)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            mask = mask.to(h.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (h * mask).sum(dim=1) / denom
        return self.rho(pooled)

# ─── Load & preprocess ───────────────────────────────────────────────────────
df = pd.read_csv("hour.csv")[["dteday","holiday","weekday"] + feats + ["cnt"]]
df["sqrt_cnt"] = np.sqrt(df["cnt"])
# De-mean within (holiday, weekday) to remove coarse seasonal effects
df["cnt"]      = df.groupby(["holiday","weekday"])["sqrt_cnt"].transform(
    lambda y: y - y.mean()
)

# ─── Summary ────────────────────────────────────────────────────────────────
env_sizes = df.groupby('dteday').size()
print("──────────────── Dataset summary ────────────────")
print(f"• Number of environments (days): {int(env_sizes.shape[0])}")
print(f"• Mean samples per environment: {float(env_sizes.mean()):.2f}")
print(f"• Number of features used: {len(feats)}")
print("────────────────────────────────────────────────")

# ─── Folds ───────────────────────────────────────────────────────────────────
days  = np.array(sorted(df["dteday"].unique()))
folds = [list(chunk) for chunk in np.array_split(days, n_folds)]

# ─── Feature subsets ─────────────────────────────────────────────────────────
all_subsets = [[]] + [list(c) for k in range(1,len(feats)+1)
                      for c in combinations(feats,k)]
K           = len(all_subsets)

# ─── Helpers ────────────────────────────────────────────────────────────────
def aligned_proba(estimator, x_row, K):
    """
    Return a length-K probability vector aligned to subset ids 0..K-1,
    even if some classes are missing in estimator.classes_.
    """
    p = np.zeros(K, dtype=float)
    probs = estimator.predict_proba([x_row])[0]
    for i, cls in enumerate(estimator.classes_):
        p[int(cls)] = probs[i]
    return p

# ─── Anchor ───
def fit_anchor(df_tr, g):
    days_tr = df_tr["dteday"].unique()
    Xd      = pd.get_dummies(
        pd.Categorical(df_tr["dteday"], categories=days_tr)
    ).values
    Y       = df_tr[feats+["cnt"]].values
    full  = LinearRegression(fit_intercept=False).fit(Xd, Y)
    const = LinearRegression(fit_intercept=False).fit(
        np.ones((len(df_tr),1)), Y
    )
    resid = Y - full.predict(Xd)
    newY  = const.predict(np.ones((len(df_tr),1))) \
            + resid \
            + g*(full.predict(Xd) - const.predict(np.ones((len(df_tr),1))))
    mdl   = LinearRegression().fit(newY[:,:-1], newY[:,-1])
    return full, const, mdl

def anchor_fold_loss(df_tr, df_te, g):
    full, const, mdl = fit_anchor(df_tr, g)
    days_tr          = df_tr["dteday"].unique()
    Xd_te            = pd.get_dummies(
        pd.Categorical(df_te["dteday"], categories=days_tr)
    ).values
    Y_te             = df_te[feats+["cnt"]].values
    resid_te         = Y_te - full.predict(Xd_te)
    new_te           = const.predict(np.ones((len(df_te),1))) \
                       + resid_te \
                       + g*(full.predict(Xd_te) - const.predict(np.ones((len(df_te),1))))
    preds            = mdl.predict(new_te[:,:-1])
    df2              = pd.DataFrame(
        {"dteday": df_te["dteday"], "loss": (preds - df_te["cnt"].values)**2}
    )
    return df2.groupby("dteday")["loss"].mean().values

def tune_anchor_gamma(df_tr, inner_splits, gamma_grid):
    """Select Anchor gamma via inner CV"""
    days_tr = df_tr["dteday"].unique()
    scores = []
    for g in gamma_grid:
        vals = []
        for train_idx, val_idx in inner_splits:
            train_days = days_tr[train_idx]
            valid_days = days_tr[val_idx]
            tr_in = df_tr[df_tr["dteday"].isin(train_days)]
            va_in = df_tr[df_tr["dteday"].isin(valid_days)]
            vals.append(float(anchor_fold_loss(tr_in, va_in, g).mean()))
        scores.append(float(np.mean(vals)))

    best_idx = int(np.argmin(scores))
    return float(gamma_grid[best_idx]), float(scores[best_idx])

# ─── Env summary stats ───────────────────────────────────────────────────────
def env_summary_stats(env, cols):
    X = env[cols].values
    n, p = X.shape
    means = [np.mean(X[:, j]) if n >= 1 else 0 for j in range(p)]
    sds   = [np.std(X[:, j], ddof=1) if n >= 2 else 0 for j in range(p)]
    if n >= 2:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        S  = np.cov(Xc.T)
        try:
            Omega = la.inv(S)
        except la.LinAlgError:
            Omega = la.pinv(S)
        pcorrs = []
        for i in range(p):
            for j in range(i+1,p):
                denom = Omega[i,i] * Omega[j,j]
                if denom <= 0 or not np.isfinite(denom):
                    pcorrs.append(0.0)
                else:
                    val = -Omega[i,j] / np.sqrt(denom)
                    pcorrs.append(val if np.isfinite(val) else 0.0)
    else:
        pcorrs = [0.0]*(p*(p-1)//2)
    return np.array(means + sds + pcorrs, dtype=float)

# ─── DeepSet helpers ────────────────────────────────────────────────────────
def build_env_batch(envs_list, idxs, Nmax=None):
    sets = [torch.tensor(envs_list[i][1][feats].values,
                         dtype=torch.float32)
            for i in idxs]
    lengths = [x.shape[0] for x in sets]

    if Nmax is None:
        Nmax = max(lengths) if lengths else 1

    if not sets:
        Xbat = torch.zeros((0, Nmax, len(feats)), dtype=torch.float32)
        mask = torch.zeros((0, Nmax, 1), dtype=torch.float32)
        return Xbat, mask, Nmax

    Xbat = torch.stack([F.pad(x, (0, 0, 0, Nmax - x.shape[0])) for x in sets])
    mask = torch.stack([
        F.pad(torch.ones((x.shape[0], 1), dtype=torch.float32),
              (0, 0, 0, Nmax - x.shape[0]))
        for x in sets
    ])
    return Xbat, mask, Nmax

def train_deep_set_model(cfg, envs_list, y_labels, train_idx,
                         val_idx=None, max_epochs=30, patience=5):
    y_train = torch.tensor(y_labels[train_idx], device=DEVICE)
    X_train, M_train, Nmax = build_env_batch(envs_list, train_idx)
    X_train = X_train.to(DEVICE)
    M_train = M_train.to(DEVICE)

    ds   = EnvDeepSetEmbed(len(feats), cfg["phi_hidden"],
                           cfg["rho_hidden"], cfg["embed_dim"]).to(DEVICE)
    head = nn.Linear(cfg["embed_dim"], K).to(DEVICE)
    opt  = optim.Adam(list(ds.parameters())+list(head.parameters()),
                      lr=cfg["lr"], weight_decay=cfg["wd"])

    has_val = val_idx is not None and len(val_idx) > 0
    if has_val:
        y_val = torch.tensor(y_labels[val_idx], device=DEVICE)
        X_val, M_val, _ = build_env_batch(envs_list, val_idx, Nmax=Nmax)
        X_val = X_val.to(DEVICE)
        M_val = M_val.to(DEVICE)

    best_state = None
    best_val = float('inf')
    no_improve = 0

    for _ in range(max_epochs):
        ds.train(); head.train()
        opt.zero_grad()
        logits = head(ds(X_train, M_train))
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

        if has_val:
            ds.eval(); head.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(head(ds(X_val, M_val)), y_val).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = (ds.state_dict(), head.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if has_val and best_state is not None:
        ds.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])

    return ds, head, Nmax

def mixture_mse(env_df, selector, baseline_models, name,
                ds_mod=None, ds_head=None, Nmax=None, summ_scaler=None):
    if name == "DS":
        x = torch.tensor(env_df[feats].values,
                         dtype=torch.float32).to(DEVICE)
        n = x.shape[0]
        pad = F.pad(x, (0, 0, 0, Nmax - n)).unsqueeze(0)
        mask = torch.ones((n, 1), dtype=torch.float32, device=DEVICE)
        mask = F.pad(mask, (0, 0, 0, Nmax - n)).unsqueeze(0)
        probs = torch.softmax(ds_head(ds_mod(pad, mask)), dim=1).detach().cpu().numpy()[0]
    else:
        stats = env_summary_stats(env_df, feats)
        if summ_scaler is not None:
            stats = summ_scaler.transform(stats.reshape(1,-1))[0]
        probs = aligned_proba(selector, stats, K)
    mix = np.zeros(len(env_df))
    for k, pk in enumerate(probs):
        if pk > 0:
            sub  = all_subsets[k]
            Xsub = env_df[list(sub)].values if sub else np.zeros((len(env_df),0))
            mix += pk * baseline_models[tuple(sub)].predict(Xsub)
    return mean_squared_error(env_df["cnt"].values, mix)

# ─── ICP invariance test ─────────────────────────────────────────────────
def _rss_qr(X: np.ndarray, y: np.ndarray):
    """Least-squares fit and return RSS and rank."""
    beta, _, rank, _ = la.lstsq(X, y, lapack_driver="gelsy")
    resid = y - X @ beta
    rss = float(resid @ resid)
    return rss, int(rank)

def icp_invariant(tr_data, alpha):
    # Dummy variables for days (anchors)
    dums = pd.get_dummies(
        tr_data["dteday"], prefix="day",
        drop_first=True, dtype=float
    ).to_numpy()
    y = tr_data["cnt"].to_numpy(dtype=float)
    passing = []
    n = len(tr_data)
    for sub in all_subsets[1:]:
        X0 = np.column_stack([
            np.ones(n, dtype=float),
            tr_data[list(sub)].to_numpy(dtype=float)
        ])
        X_aug = np.column_stack([X0, dums]) if dums.size else X0.copy()
        rss0, r0 = _rss_qr(X0,  y)
        rss1, r1 = _rss_qr(X_aug, y)
        df1 = r1 - r0
        df2 = n  - r1
        if df1 <= 0 or df2 <= 0:
            pval = 0.0
        else:
            num = max(rss0 - rss1, 0.0)
            den = rss1 / df2 if rss1 > 0 else 0.0
            Fval = (num / df1) / den if den > 0 else np.inf
            pval = float(fdist.sf(Fval, df1, df2))
        if pval > alpha:
            passing.append(set(sub))
    return sorted(set.intersection(*passing)) if passing else []

# ─── Plot helpers ────────────────────────────────────────────────────────────
def _clean_series_for_plot(cv_mses: dict) -> dict:
    # Keep only EACS (Best) among EACS variants
    cv = {k: v for k, v in cv_mses.items()
          if not (k.startswith('EACS (') and k != 'EACS (Best)')}
    cleaned = {}
    for k, v in cv.items():
        arr = np.array([x for x in v if np.isfinite(x)], dtype=float)
        if arr.size > 0:
            cleaned[k] = arr
    return cleaned

def plot_results(cv_mses):
    """Render the final summary plot."""
    cv_mses = _clean_series_for_plot(cv_mses)

    methods = list(cv_mses.keys())
    means = np.array([np.mean(v) for v in cv_mses.values()], dtype=float)
    stds  = np.array([np.std(v)  for v in cv_mses.values()], dtype=float)

    print("\nSummary statistics:")
    for m, mu, sd in zip(methods, means, stds):
        print(f"  {m}: {mu:.4f} ± {sd:.4f}")

    def _display_label(m: str) -> str:
        if m == "EACS (Best)":
            return "EACS"
        if m.endswith("(baseline)"):
            feat_str = m.replace(" (baseline)", "")
            return f"Baseline ({feat_str})"
        return m

    def _plot_color(m: str) -> str:
        if m == "Oracle":
            return "magenta"
        if m == "EACS (Best)":
            return "green"
        if m == "Anchor":
            return "blue"
        if m == "Lasso":
            return "purple"
        if m == "ICP":
            return "red"
        if m.endswith("(baseline)"):
            return "gray"
        return "gray"

    def _tiebreak_priority(m: str) -> int:
        # Only used when mean MSE ties.
        if m == "Oracle":
            return 0
        if m == "EACS (Best)":
            return 1
        if m == "Anchor":
            return 2
        if m == "Lasso":
            return 3
        if m.endswith("(baseline)"):
            return 4
        if m == "ICP":
            return 5
        return 999

    order = sorted(
        range(len(methods)),
        key=lambda i: (float(means[i]), _tiebreak_priority(methods[i]), methods[i]),
    )
    methods = [methods[i] for i in order]
    means   = [float(means[i]) for i in order]
    stds    = [float(stds[i]) for i in order]

    y_labels = [_display_label(m) for m in methods]

    # Scale-aware x-offset for the numeric annotations
    xmin = min(mu - sd for mu, sd in zip(means, stds))
    xmax = max(mu + sd for mu, sd in zip(means, stds))
    xoff = max(0.002, 0.02 * (xmax - xmin))

    plt.figure(figsize=(10, 7.5))
    y_pos = np.arange(len(methods))

    for i, m in enumerate(methods):
        c = _plot_color(m)
        plt.errorbar(
            means[i],
            y_pos[i],
            xerr=stds[i],
            fmt="o",
            capsize=5,
            color=c,
            ecolor=c,
        )
        plt.text(
            means[i] + xoff,
            y_pos[i],
            f"{means[i]:.3f} ({stds[i]:.3f})",
            fontsize=9,
            va="bottom",
        )

    plt.yticks(y_pos, y_labels)
    plt.xlabel("MSE (±1 SD)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def try_load_and_plot():
    if not os.path.exists(PICKLE_PATH):
        return False
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)
    plot_results(data["cv_mses"])
    return True

if try_load_and_plot():
    sys.exit(0)

# ─── Outer-fold evaluation pipeline ──
nested_mses = []
nested_gs = []
nested_mses_inner = []

oracle_mses            = []
baseline_mses          = {tuple(sub): [] for sub in all_subsets}
lasso_mses             = []
best_eacs_p            = []  
best_lr_mses           = []
best_nn_mses           = []
best_rf_mses           = []
best_ds_mses           = []
best_eacs_outer_mses    = []
best_lr_outer_mses     = []
best_nn_outer_mses     = []
best_rf_outer_mses     = []
best_ds_outer_mses     = []
icp_tuned_mses         = []

for i, te_days in enumerate(folds, start=1):
    tr = df[~df["dteday"].isin(te_days)].copy()
    te = df[df["dteday"].isin(te_days)].copy()

    # Standardize covariates for ALL methods in this outer fold
    fold_scaler = StandardScaler().fit(tr[feats].values.astype(float))
    tr.loc[:, feats] = fold_scaler.transform(tr[feats].values.astype(float))
    te.loc[:, feats] = fold_scaler.transform(te[feats].values.astype(float))

    days_tr   = tr["dteday"].unique()
    skf_inner = KFold(n_splits=INNER_CV, shuffle=True, random_state=SEED)
    inner_splits_days = list(skf_inner.split(days_tr))

    # 0) Anchor (tune γ on outer-train; evaluate on outer-test)
    g, inner_mse = tune_anchor_gamma(tr, inner_splits_days, gamma_grid)
    outer_mse = float(anchor_fold_loss(tr, te, g).mean())
    nested_gs.append(g)
    nested_mses_inner.append(inner_mse)
    nested_mses.append(outer_mse)

    # 1) Baselines & Oracle
    baseline_cv = {}
    for sub in all_subsets:
        Xtr = tr[list(sub)].values if sub else np.zeros((len(tr),0))
        mdl = DummyRegressor(strategy="mean") if not sub else LinearRegression()
        baseline_cv[tuple(sub)] = mdl.fit(Xtr, tr["cnt"].values)

    fos = []
    for _,env in te.groupby("dteday"):
        ms = [mean_squared_error(
                env["cnt"].values,
                baseline_cv[tuple(sub)].predict(
                    env[list(sub)].values if sub else np.zeros((len(env),0))
                )
              )
              for sub in all_subsets]
        fos.append(min(ms))
    oracle_mses.append(np.mean(fos))

    for sub, mdl in baseline_cv.items():
        dms = []
        for _, env in te.groupby("dteday"):
            preds = mdl.predict(
                env[list(sub)].values if sub else np.zeros((len(env),0))
            )
            dms.append(mean_squared_error(env["cnt"].values, preds))
        baseline_mses[sub].append(np.mean(dms))

    # 1b) Lasso
    lasso_alphas   = [0.001, 0.01, 0.1, 1.0]
    best_alpha     = None
    best_lasso_mse = np.inf
    for alpha_ in lasso_alphas:
        fold_mses = []
        for train_idx, val_idx in skf_inner.split(days_tr):
            train_days = days_tr[train_idx]
            val_days   = days_tr[val_idx]
            tr_in      = tr[tr["dteday"].isin(train_days)]
            va_in      = tr[tr["dteday"].isin(val_days)]
            mdl = Lasso(alpha=alpha_, max_iter=10000)
            mdl.fit(tr_in[feats].values, tr_in["cnt"].values)
            day_mses = []
            for d in val_days:
                env_preds = mdl.predict(va_in[va_in["dteday"] == d][feats].values)
                env_true  = va_in[va_in["dteday"] == d]["cnt"].values
                day_mses.append(mean_squared_error(env_true, env_preds))
            fold_mses.append(np.mean(day_mses))
        inner_mse_lasso = np.mean(fold_mses)
        if inner_mse_lasso < best_lasso_mse:
            best_lasso_mse, best_alpha = inner_mse_lasso, alpha_
    model_lasso = Lasso(alpha=best_alpha, max_iter=10000)
    model_lasso.fit(tr[feats].values, tr["cnt"].values)
    lasso_mses.append(
        np.mean([
            mean_squared_error(env["cnt"].values,
                               model_lasso.predict(env[feats].values))
            for _, env in te.groupby("dteday")
        ])
    )

    # 2) Env summaries & oracle labels on outer‑train
    envs_list = list(tr.groupby("dteday"))
    X_env_all = np.vstack([env_summary_stats(env, feats) for _,env in envs_list])

    # Standardize env summaries across training environments
    summ_scaler = StandardScaler().fit(X_env_all.astype(float))
    X_env_all   = summ_scaler.transform(X_env_all.astype(float))

    y_env_all = np.array([
        int(np.argmin([
            mean_squared_error(
                env_df["cnt"].values,
                baseline_cv[tuple(sub)].predict(
                    env_df[list(sub)].values if sub else np.zeros((len(env_df), 0)
                ))
            )
            for sub in all_subsets
        ]))
        for _, env_df in envs_list
    ])
    env_ids = np.arange(len(envs_list))

    inner = KFold(n_splits=INNER_CV, shuffle=True, random_state=SEED)

    # Helper: evaluate a selector (hard/soft) via inner CV
    def eval_selector(estimator, mode='hard'):
        mses = []
        for tr_idx, va_idx in inner.split(env_ids):
            est = clone(estimator)
            est.fit(X_env_all[tr_idx], y_env_all[tr_idx])
            for j in va_idx:
                env_df = envs_list[j][1]
                if mode == 'hard':
                    k = est.predict(X_env_all[j:j+1])[0]
                    sub = all_subsets[int(k)]
                    Xsub = env_df[list(sub)].values if sub else np.zeros((len(env_df),0))
                    pred = baseline_cv[tuple(sub)].predict(Xsub)
                    mses.append(mean_squared_error(env_df["cnt"].values, pred))
                else:
                    mses.append(
                        mixture_mse(env_df, est, baseline_cv, "SEL",
                                    summ_scaler=summ_scaler)
                    )
        return float(np.mean(mses))

    # 3) Logistic Regression
    C_grid = [0.1, 1.0, 10.0]
    lr_best = {'mse': np.inf, 'C': None, 'mode': None}
    for C in C_grid:
        base = LogisticRegression(
            C=C,
            penalty="l2",
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced"
        )
        mse_h = eval_selector(base, mode='hard')
        mse_s = eval_selector(base, mode='soft')
        if mse_h < lr_best['mse']:
            lr_best.update({'mse': mse_h, 'C': C, 'mode': 'hard'})
        if mse_s < lr_best['mse']:
            lr_best.update({'mse': mse_s, 'C': C, 'mode': 'soft'})
    best_lr_mses.append(lr_best['mse'])

    model_lr = LogisticRegression(
        C=lr_best['C'],
        penalty="l2",
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced"
    )
    model_lr.fit(X_env_all, y_env_all)
    lr_mode = lr_best['mode']

    # 4) Random Forest
    rf_base = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=0,
        n_jobs=-1
    )
    def eval_rf(mode='hard'):
        mses = []
        for tr_idx, va_idx in inner.split(env_ids):
            est = clone(rf_base)
            est.fit(X_env_all[tr_idx], y_env_all[tr_idx])
            for j in va_idx:
                env_df = envs_list[j][1]
                if mode == "hard":
                    k = est.predict(X_env_all[j:j+1])[0]
                    sub = all_subsets[int(k)]
                    Xsub = env_df[list(sub)].values if sub else np.zeros((len(env_df),0))
                    pred = baseline_cv[tuple(sub)].predict(Xsub)
                    mses.append(mean_squared_error(env_df["cnt"].values, pred))
                else:
                    mses.append(
                        mixture_mse(env_df, est, baseline_cv, "SEL",
                                    summ_scaler=summ_scaler)
                    )
        return float(np.mean(mses))

    mse_rf_h = eval_rf(mode="hard")
    mse_rf_s = eval_rf(mode="soft")
    if mse_rf_h <= mse_rf_s:
        rf_mode, best_rf_mse = 'hard', mse_rf_h
    else:
        rf_mode, best_rf_mse = 'soft', mse_rf_s
    best_rf_mses.append(best_rf_mse)
    model_rf = clone(rf_base)
    model_rf.fit(X_env_all, y_env_all)

    # 5) Neural Net selector
    mlp_base = dict(hidden_layer_sizes=(64,32),
                    learning_rate_init=1e-3,
                    early_stopping=False,
                    random_state=0,
                    max_iter=2000)
    alpha_grid = [1e-5, 1e-4, 1e-3]
    nn_best = {'mse': np.inf, 'alpha': None, 'mode': None}
    for a in alpha_grid:
        base = MLPClassifier(alpha=a, **mlp_base)
        mse_h = eval_selector(base, mode='hard')
        mse_s = eval_selector(base, mode='soft')
        if mse_h < nn_best['mse']:
            nn_best.update({'mse': mse_h, 'alpha': a, 'mode': 'hard'})
        if mse_s < nn_best['mse']:
            nn_best.update({'mse': mse_s, 'alpha': a, 'mode': 'soft'})
    best_nn_mses.append(nn_best['mse'])
    model_nn = MLPClassifier(alpha=nn_best['alpha'], **mlp_base)
    model_nn.fit(X_env_all, y_env_all)
    nn_mode = nn_best['mode']

    # 6) DeepSet gate
    ds_cfg = {"phi_hidden":64, "rho_hidden":64, "embed_dim":32,
              "lr":1e-3, "wd":0.0}
    ds_fold_hard, ds_fold_soft = [], []
    for tr_idx, va_idx in inner.split(env_ids):
        ds_mod, ds_head, Nmax_tr = train_deep_set_model(
            ds_cfg, envs_list, y_env_all,
            train_idx=tr_idx, val_idx=va_idx,
            max_epochs=30, patience=5
        )
        for j in va_idx:
            env_df = envs_list[j][1]
            # hard
            x = torch.tensor(env_df[feats].values,
                             dtype=torch.float32).to(DEVICE)
            n = x.shape[0]
            pad = F.pad(x, (0, 0, 0, Nmax_tr - n)).unsqueeze(0)
            mask = torch.ones((n, 1), dtype=torch.float32, device=DEVICE)
            mask = F.pad(mask, (0, 0, 0, Nmax_tr - n)).unsqueeze(0)
            k = int(torch.argmax(ds_head(ds_mod(pad, mask)), dim=1).item())
            sub = all_subsets[k]
            Xsub = env_df[list(sub)].values if sub else np.zeros((len(env_df),0))
            pred = baseline_cv[tuple(sub)].predict(Xsub)
            ds_fold_hard.append(
                mean_squared_error(env_df["cnt"].values, pred)
            )
            # soft
            ds_fold_soft.append(
                mixture_mse(env_df, None, baseline_cv, "DS",
                            ds_mod=ds_mod, ds_head=ds_head,
                            Nmax=Nmax_tr)
            )
    ds_mse_h = float(np.mean(ds_fold_hard))
    ds_mse_s = float(np.mean(ds_fold_soft))
    if ds_mse_h <= ds_mse_s:
        ds_mode, best_ds_mse = 'hard', ds_mse_h
    else:
        ds_mode, best_ds_mse = 'soft', ds_mse_s
    best_ds_mses.append(best_ds_mse)

    ds_model, ds_head, Nmax_ds = train_deep_set_model(
        ds_cfg, envs_list, y_env_all,
        train_idx=env_ids,
        val_idx=None,
        max_epochs=30, patience=5
    )

    # Outer test losses per selector family
    def compute_losses(model, tag, env_iter, mode):
        det, soft = [], []
        for _, env in env_iter:
            stats = env_summary_stats(env, feats)
            stats_scaled = summ_scaler.transform(stats.reshape(1, -1))[0]
            if mode == "hard":
                k = int(model.predict([stats_scaled])[0])
                sub = all_subsets[k]
                Xsub = env[list(sub)].values if sub else np.zeros((len(env),0))
                det.append(
                    mean_squared_error(
                        env["cnt"].values,
                        baseline_cv[tuple(sub)].predict(Xsub)
                    )
                )
                soft.append(
                    mixture_mse(env, model, baseline_cv, tag,
                                summ_scaler=summ_scaler)
                )
            else:  # mode == "soft"
                soft.append(
                    mixture_mse(env, model, baseline_cv, tag,
                                summ_scaler=summ_scaler)
                )
        if mode == "hard":
            return float(np.mean(det))
        else:
            return float(np.mean(soft))

    lr_outer = compute_losses(model_lr, "LR",
                              te.groupby("dteday"), lr_mode)
    nn_outer = compute_losses(model_nn, "NN",
                              te.groupby("dteday"), nn_mode)
    rf_outer = compute_losses(model_rf, "RF",
                              te.groupby("dteday"), rf_mode)

    # DeepSet outer
    det_ds, soft_ds = [], []
    for _, env in te.groupby("dteday"):
        x = torch.tensor(env[feats].values,
                         dtype=torch.float32).to(DEVICE)
        n = x.shape[0]
        pad = F.pad(x, (0, 0, 0, Nmax_ds - n)).unsqueeze(0)
        mask = torch.ones((n, 1), dtype=torch.float32, device=DEVICE)
        mask = F.pad(mask, (0, 0, 0, Nmax_ds - n)).unsqueeze(0)
        k = int(torch.argmax(ds_head(ds_model(pad, mask)), dim=1).item())
        sub = all_subsets[k]
        Xsub = env[list(sub)].values if sub else np.zeros((len(env),0))
        det_ds.append(
            mean_squared_error(
                env["cnt"].values,
                baseline_cv[tuple(sub)].predict(Xsub)
            )
        )
        soft_ds.append(
            mixture_mse(env, None, baseline_cv, "DS",
                        ds_mod=ds_model, ds_head=ds_head,
                        Nmax=Nmax_ds)
        )
    ds_outer = float(np.mean(det_ds)) if ds_mode=="hard" else float(np.mean(soft_ds))

    best_lr_outer_mses.append(lr_outer)
    best_nn_outer_mses.append(nn_outer)
    best_rf_outer_mses.append(rf_outer)
    best_ds_outer_mses.append(ds_outer)

    # EACS winner for this fold (by inner CV MSE)
    inner_candidates = {
        "LR": lr_best['mse'],
        "NN": nn_best['mse'],
        "RF": best_rf_mse,
        "DS": best_ds_mse
    }
    outers           = {
        "LR": lr_outer,
        "NN": nn_outer,
        "RF": rf_outer,
        "DS": ds_outer
    }
    best_method = min(inner_candidates, key=inner_candidates.get)
    best_outer  = outers[best_method]
    best_eacs_outer_mses.append(best_outer)

    # ── ICP tuning over thresholds ─────────────────────────────────────────
    thr_grid = np.array([0.01, 0.05, 0.10])
    best_thr, best_mse = None, np.inf
    for thr in thr_grid:
        mvals = []
        for train_idx, val_idx in inner_splits_days:
            train_days = days_tr[train_idx]
            valid_days = days_tr[val_idx]
            tr_in = tr[tr["dteday"].isin(train_days)]
            va_in = tr[tr["dteday"].isin(valid_days)]
            inv = icp_invariant(tr_in, thr)
            mdl = DummyRegressor() if not inv else LinearRegression()
            Xtr_i = tr_in[list(inv)].values if inv else np.zeros((len(tr_in),0))
            mdl.fit(Xtr_i, tr_in["cnt"].values)
            loss = []
            for _, env in va_in.groupby("dteday"):
                Xv = env[list(inv)].values if inv else np.zeros((len(env),0))
                loss.append(
                    mean_squared_error(env["cnt"].values, mdl.predict(Xv))
                )
            mvals.append(np.mean(loss))
        m = np.mean(mvals)
        if m < best_mse:
            best_mse, best_thr = m, thr

    inv = icp_invariant(tr, best_thr)
    mdl = DummyRegressor() if not inv else LinearRegression()
    Xtr_i = tr[list(inv)].values if inv else np.zeros((len(tr),0))
    mdl.fit(Xtr_i, tr["cnt"].values)
    losses = []
    for _,env in te.groupby("dteday"):
        Xv = env[list(inv)].values if inv else np.zeros((len(env),0))
        losses.append(mean_squared_error(env["cnt"].values, mdl.predict(Xv)))
    icp_mse = float(np.mean(losses))
    icp_tuned_mses.append(icp_mse)

    # ─── Fold summary logging ─────────────────────────────────────────────
    lasso_hp = f"α={best_alpha}"
    lr_hp    = f"C={lr_best['C']}"
    nn_hp    = f"alpha={nn_best['alpha']}, layers=(64,32), lr=1e-3"
    rf_hp    = ("n_estimators=100, max_depth=None, "
                "min_samples_leaf=1, max_features='sqrt'")
    ds_hp    = (f"phi={ds_cfg['phi_hidden']}, rho={ds_cfg['rho_hidden']}, "
                f"embed_dim={ds_cfg['embed_dim']}, lr={ds_cfg['lr']}, "
                f"wd={ds_cfg['wd']}")
    print(f"\n--- Fold {i} summary ---")
    print(f"Anchor: γ={g:.4f}, outer‑CV MSE={outer_mse:.4f}")
    print(f"Oracle: outer‑CV MSE={oracle_mses[-1]:.4f}")
    print(f"Lasso:  {lasso_hp}, outer‑CV MSE={lasso_mses[-1]:.4f}")
    print(f"EACS (LR): mode={lr_mode}, {lr_hp}, outer‑CV MSE={lr_outer:.4f}, "
          f"inner‑CV MSE={lr_best['mse']:.4f}")
    print(f"EACS (NN): mode={nn_mode}, {nn_hp}, outer‑CV MSE={nn_outer:.4f}, "
          f"inner‑CV MSE={nn_best['mse']:.4f}")
    print(f"EACS (RF): mode={rf_mode}, {rf_hp}, outer‑CV MSE={rf_outer:.4f}, "
          f"inner‑CV MSE={best_rf_mse:.4f}")
    print(f"EACS (DS): mode={ds_mode}, {ds_hp}, outer‑CV MSE={ds_outer:.4f}, "
          f"inner‑CV MSE={best_ds_mse:.4f}")
    print(f"EACS (Best): method={best_method}, outer‑CV MSE={best_outer:.4f}")
    print(f"ICP:    τ={best_thr:.3f}, outer‑CV MSE={icp_mse:.4f}")

# ─── Plot + save ────────────────────────────────────────────────────────────
cv_mses = {
    'Oracle':        oracle_mses,
    'Anchor':        nested_mses,
    'Lasso':         lasso_mses,
    'EACS (LR)':      best_lr_outer_mses,
    'EACS (NN)':      best_nn_outer_mses,
    'EACS (RF)':      best_rf_outer_mses,
    'EACS (DS)':      best_ds_outer_mses,
    'EACS (Best)':    best_eacs_outer_mses,
    'ICP':           icp_tuned_mses
}
for sub, vals in baseline_mses.items():
    label = ", ".join(sub) if len(sub) > 0 else "intercept-only"
    cv_mses[f"{label} (baseline)"] = vals

to_save = {'cv_mses': cv_mses}
with open(PICKLE_PATH,'wb') as f:
    pickle.dump(to_save, f)
print("➡️ Computed and saved metrics to", PICKLE_PATH)

plot_results(cv_mses)

