# ─────────────────────────────────────────────────────────────────────────────
# ACS Income analysis 
# ─────────────────────────────────────────────────────────────────────────────

import os

_N_THREADS = os.environ.get("N_THREADS", str(os.cpu_count() or 1))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _N_THREADS)

import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(int(_N_THREADS))

# ─── Seed / device ───────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data loading / preprocessing ────────────────────────────────────────────
DATA_PATH = "preprocessed_income_2018.csv"
# Output paths (one run writes all of these).
PICKLE_PATH = "income_res.pkl"
FIG_PATH = "income_res.png"
ALL_RESULTS_CSV_PATH = "income_all_results_summary.csv"
SUPP_TABLE_CSV_PATH = "income_env_aug_supp_table.csv"
SUPP_TABLE_TEX_PATH = "income_env_aug_supp_table.tex"
ENV_COL = "STATE"
TARGET_COL = "LogIncome"


df = pd.read_csv(DATA_PATH)

# Load data

# Categorical recoding helper
def binarize_data(df: pd.DataFrame):
    """Map categorical codes to bins, then one-hot, then cast to int."""
    ACSIncome_categories = {
        "COW": {
            1.0: "Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions",
            2.0: "Employee of a private not-for-profit, tax-exempt, or charitable organization",
            3.0: "Local government employee (city, county, etc.)",
            4.0: "State government employee",
            5.0: "Federal government employee",
            6.0: "Self-employed in own not incorporated business, professional practice, or farm",
            7.0: "Self-employed in own incorporated business, professional practice or farm",
            8.0: "Working without pay in family business or farm",
            9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
        },
        "SCHL": {
            **{float(i): "No College degree" for i in range(1, 21)},
            21.0: "College degree",
            22.0: "Higher",
            23.0: "Higher",
            24.0: "Higher",
        },
        "MAR": {
            1.0: "Married",
            2.0: "Not Married",
            3.0: "Not Married",
            4.0: "Not Married",
            5.0: "Not Married",
        },
        "SEX": {
            1.0: "Male",
            2.0: "Female",
        },
        "RAC1P": {
            1.0: "White",
            2.0: "Black or African American",
            3.0: "Other",
            4.0: "Other",
            5.0: "Other",
            6.0: "Asian",
            7.0: "Other",
            8.0: "Other",
            9.0: "Other",
        },
    }

    df = df.copy()
    for col, mapping in ACSIncome_categories.items():
        df[col] = df[col].map(mapping)

    cat_cols = list(ACSIncome_categories.keys())
    dummies = pd.get_dummies(df[cat_cols].astype("category"), drop_first=False)
    dummies = dummies.astype(int)

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, dummies], axis=1)

    return df

df = binarize_data(df)

# Features
drop_cols = [ENV_COL, TARGET_COL]
feats = [c for c in df.columns if c not in drop_cols]
p_feat = len(feats)

# ─── Env folds ───────────────────────────────────────────────────────────────
n_folds = 5
days = np.array(sorted(df[ENV_COL].unique()))
fold_sz = np.linspace(0, len(days), n_folds + 1, dtype=int)
folds = [days[fold_sz[i] : fold_sz[i + 1]] for i in range(n_folds)]

# ─── Per-environment summaries ─────────────────────────────────────────────
def env_summary_stats_from_X(X: np.ndarray, alpha_max: float = 0.3) -> np.ndarray:
    """Per-environment summary vector from standardized covariates.
    
    Returns concatenated (means, stds, partial correlations).
    """

    X = np.asarray(X, dtype=float)
    n, p = X.shape

    means = X.mean(axis=0) if n >= 1 else np.zeros(p, dtype=float)
    sds   = X.std(axis=0, ddof=1) if n >= 2 else np.zeros(p, dtype=float)

    # Partial correlations
    if n >= 2 and p >= 2:
        with np.errstate(all="ignore"):
            cov = np.cov(X, rowvar=False)
        alpha = min(float(alpha_max), float(p) / float(max(n, p + 1)))
        cov_shrink = (1.0 - alpha) * cov + alpha * np.diag(np.diag(cov))

        try:
            prec = np.linalg.inv(cov_shrink)
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(cov_shrink)

        pcs = []
        for i in range(p):
            pii = float(prec[i, i])
            for j in range(i + 1, p):
                pjj = float(prec[j, j])
                denom = pii * pjj
                if denom <= 0 or not np.isfinite(denom):
                    pcs.append(0.0)
                    continue
                val = -float(prec[i, j]) / float(np.sqrt(denom))
                pcs.append(val if np.isfinite(val) else 0.0)
        pcs = np.asarray(pcs, dtype=float)
    else:
        pcs = np.zeros(p * (p - 1) // 2, dtype=float)

    return np.concatenate([means, sds, pcs], axis=0).astype(float)
# ─── Cache per-env (X, y) ───────────────────────────────────────────────────
# Keep raw (unstandardized) covariates; outer-fold standardization is applied later.
X_cache_orig, y_cache = {}, {}
for day, env in df.groupby(ENV_COL):
    X_cache_orig[day] = torch.tensor(env[feats].values,
                                     dtype=torch.float32, device=DEVICE)
    y_cache[day] = torch.tensor(env[TARGET_COL].values,
                                dtype=torch.float32, device=DEVICE)

# Summary dimension: |u_e| = 2p + p(p-1)/2  (means, sds, partial corrs)
dim_u = int(2 * p_feat + (p_feat * (p_feat - 1)) // 2)
# ─── Models ─────────────────────────────────────────────────────────────────
class EnvDeepSetEmbed(nn.Module):
    def __init__(self, in_dim, phi_hidden=128, rho_hidden=128, embed_dim=64):
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

    def forward(self, X):
        h = self.phi(X)
        h_mean = h.mean(dim=1)
        return self.rho(h_mean)

class GateMLPLogits(nn.Module):
    def __init__(self, input_dim, p_feat, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p_feat),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, ctx):
        return self.mlp(ctx)

def soft_mask_from_logits(a: torch.Tensor, temp: float) -> torch.Tensor:
    """Return sigmoid(a/temp)."""
    return torch.sigmoid(a / float(temp))


class EnvAugLinearInteraction(nn.Module):
    """Penalized linear predictor on X, u_e, and X ⊗ u_e without materializing interactions.

    y_hat = b + gamma^T u_e + X_i^T (beta + B u_e)

    This is the scalable ACS analogue of a ridge regression on [X, u_e, X ⊗ u_e].
    """

    def __init__(self, p: int, q: int, bias_init: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias_init), dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(p, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(q, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(p, q, dtype=torch.float32))

    def forward_env(self, X: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 2:
            u = u.squeeze(0)
        env_coef = self.beta + torch.mv(self.B, u)
        env_bias = self.bias + torch.dot(self.gamma, u)
        return env_bias + torch.mv(X, env_coef)

    def ridge_penalty(self) -> torch.Tensor:
        # Mean-squared penalty keeps the tuning scale stable as p and q vary.
        return (
            self.beta.pow(2).mean()
            + self.gamma.pow(2).mean()
            + self.B.pow(2).mean()
        )

# ─── Anchor regression ──────────────────────────────────────────────────────
def fit_anchor(df_tr: pd.DataFrame, g: float, features):
    days_tr = df_tr[ENV_COL].unique()
    Xd = pd.get_dummies(
        pd.Categorical(df_tr[ENV_COL], categories=days_tr)
    ).values
    Y = df_tr[features + [TARGET_COL]].values

    full  = LinearRegression().fit(Xd, Y)
    const = LinearRegression().fit(np.ones((len(df_tr), 1)), Y)

    resid = Y - full.predict(Xd)
    Y0    = const.predict(np.ones((len(df_tr), 1)))
    newY  = Y0 + resid + g * (full.predict(Xd) - Y0)

    mdl = LinearRegression().fit(newY[:, :-1], newY[:, -1])
    return full, const, mdl

def anchor_fold_loss(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                     g: float, features):
    full, const, mdl = fit_anchor(df_tr, g, features)

    days_tr = df_tr[ENV_COL].unique()
    Xd_te = pd.get_dummies(
        pd.Categorical(df_te[ENV_COL], categories=days_tr)
    ).values
    Y_te = df_te[features + [TARGET_COL]].values

    resid_te = Y_te - full.predict(Xd_te)
    Y0_te    = const.predict(np.ones((len(df_te), 1)))
    new_te   = Y0_te + resid_te + g * (full.predict(Xd_te) - Y0_te)

    preds = mdl.predict(new_te[:, :-1])
    return (
        pd.DataFrame({ENV_COL: df_te[ENV_COL],
                      "loss": (preds - df_te[TARGET_COL].values) ** 2})
        .groupby(ENV_COL)["loss"]
        .mean()
        .values
    )

def tune_anchor_gamma(df_tr: pd.DataFrame,
                      inner_splits,
                      gamma_grid: np.ndarray,
                      features):
    """Select Anchor gamma via inner CV"""
    days_tr = df_tr[ENV_COL].unique()
    scores = []
    for g in gamma_grid:
        vals = []
        for tr_idx, val_idx in inner_splits:
            dtr  = days_tr[tr_idx]
            dval = days_tr[val_idx]
            tr_in = df_tr[df_tr[ENV_COL].isin(dtr)]
            va_in = df_tr[df_tr[ENV_COL].isin(dval)]
            vals.append(float(anchor_fold_loss(tr_in, va_in, g, features).mean()))
        scores.append(float(np.mean(vals)))

    best_idx = int(np.argmin(scores))
    return float(gamma_grid[best_idx]), float(scores[best_idx])

# ─── Training setup / HP grids ──────────────────────────────────────────────
INNER_FOLDS = 3
TEMP = 0.20
LAMS = [0.0]      # lam=0 (no explicit sparsity)
HID  = 128

# Budgets
HYPER_WARMUP = 10
HYPER_EPOCHS = 30
FINAL_WARMUP = 40
FINAL_EPOCHS = 120

# Opt settings
HEAD_LR = 5e-2
GATE_LR = 1e-3
CLIP_NORM = 1.0

# Anchor gamma grid
gamma_grid = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

# Environment-augmented linear interaction baseline.
RIDGE_INT_L2_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
RIDGE_INT_LR = 1e-2
RIDGE_INT_HYPER_EPOCHS = 60
RIDGE_INT_FINAL_EPOCHS = 120

# ─── Main training/eval ─────────────────────────────────────────────────────
ols_mses = []
lasso_alphas, lasso_mses = [], []
anchor_gammas, anchor_mses = [], []
ridge_int_l2s, ridge_int_inner_mses, ridge_int_mses = [], [], []
eacs_cfgs, eacs_mses = [], []  # unified EACS results

COLOR_MAP = {
    "EACS": "green",
    "Anchor": "blue",
    "OLS": "gray",
    "Lasso": "purple",
    "Ridge-Int": "black",
}

_PLOT_TIEBREAK = {
    "EACS": 0,
    "Anchor": 1,
    "OLS": 2,
    "Lasso": 3,
    "Ridge-Int": 4,
}

def _plot_order(methods, means):
    return sorted(
        range(len(methods)),
        key=lambda i: (float(means[i]), _PLOT_TIEBREAK.get(methods[i], 999)),
    )

def plot_income_results(methods, means, stds):
    """Render the final summary plot."""
    order = _plot_order(methods, means)
    methods = [methods[i] for i in order]
    means   = [float(means[i]) for i in order]
    stds    = [float(stds[i]) for i in order]

    display_methods = methods

    xmin = min(mu - sd for mu, sd in zip(means, stds))
    xmax = max(mu + sd for mu, sd in zip(means, stds))
    xoff = max(0.002, 0.02 * (xmax - xmin))

    plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(methods))
    for i, m in enumerate(methods):
        c = COLOR_MAP.get(m, 'black')
        plt.errorbar(
            means[i],
            y_pos[i],
            xerr=stds[i],
            fmt='o',
            capsize=5,
            color=c,
            ecolor=c,
        )
        plt.text(
            means[i] + xoff,
            y_pos[i],
            f"{means[i]:.3f} ({stds[i]:.3f})",
            fontsize=13,
            va='bottom',
        )

    plt.yticks(y_pos, display_methods, fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel("MSE (±1 SD)", fontsize=15)
    ax = plt.gca()
    margin = 0.4  
    ax.set_ylim(y_pos[-1] + margin, y_pos[0] - margin) 
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    print("➡️ Wrote main figure to", FIG_PATH)
    plt.show()

def print_summary(methods, means, stds):
    print("\nSummary statistics:")
    for m, mu, sd in zip(methods, means, stds):
        print(f"  {m}: {mu:.4f} ± {sd:.4f}")


def _train_ridge_interaction_model(train_days,
                                   X_src,
                                   y_src,
                                   u_src,
                                   l2: float,
                                   epochs: int,
                                   lr: float,
                                   bias_init: float) -> EnvAugLinearInteraction:
    """Train the ACS env-augmented linear interaction baseline."""
    model = EnvAugLinearInteraction(p_feat, dim_u, bias_init=bias_init).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    train_days = list(train_days)

    for _ in range(int(epochs)):
        opt.zero_grad()
        total_envs = len(train_days)
        loss = 0.0
        for d in train_days:
            yhat = model.forward_env(X_src[d], u_src[d])
            loss = loss + (1.0 / total_envs) * F.mse_loss(yhat, y_src[d])
        loss = loss + float(l2) * model.ridge_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
    return model


def _eval_ridge_interaction_model(model: EnvAugLinearInteraction,
                                  eval_days,
                                  X_src,
                                  y_src,
                                  u_src) -> float:
    """Environment-averaged MSE for the env-augmented interaction model."""
    model.eval()
    eval_days = list(eval_days)
    with torch.no_grad():
        total_envs = len(eval_days)
        se = 0.0
        for d in eval_days:
            yhat = model.forward_env(X_src[d], u_src[d])
            se += (1.0 / total_envs) * F.mse_loss(yhat, y_src[d]).item()
    return float(se)


def tune_ridge_interaction_summary(tr_days,
                                   inner_splits_days,
                                   X_src,
                                   y_src,
                                   u_src,
                                   bias_init: float):
    """Tune the penalized [X, u_e, X ⊗ u_e] baseline by environment-level CV."""
    best_l2, best_score = None, float("inf")
    tr_days = np.asarray(tr_days)
    for l2 in RIDGE_INT_L2_GRID:
        scores = []
        for tr_idx, val_idx in inner_splits_days:
            dtr = [tr_days[j] for j in tr_idx]
            dval = [tr_days[j] for j in val_idx]
            model = _train_ridge_interaction_model(
                dtr, X_src, y_src, u_src,
                l2=l2,
                epochs=RIDGE_INT_HYPER_EPOCHS,
                lr=RIDGE_INT_LR,
                bias_init=bias_init,
            )
            scores.append(
                _eval_ridge_interaction_model(model, dval, X_src, y_src, u_src)
            )
        score = float(np.mean(scores))
        if score < best_score:
            best_l2, best_score = float(l2), score

    final_model = _train_ridge_interaction_model(
        tr_days, X_src, y_src, u_src,
        l2=best_l2,
        epochs=RIDGE_INT_FINAL_EPOCHS,
        lr=RIDGE_INT_LR,
        bias_init=bias_init,
    )
    return final_model, best_l2, best_score


def _finite_stats(vals):
    arr = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return None
    return {
        "mean_mse": float(arr.mean()),
        "sd_mse": float(arr.std()),
        "n_folds": int(arr.size),
        "mse_sd": f"{arr.mean():.3f} ({arr.std():.3f})",
    }


def build_summary_df(fold_mses: dict) -> pd.DataFrame:
    rows = []
    for method, vals in fold_mses.items():
        stats = _finite_stats(vals)
        if stats is None:
            continue
        rows.append({"method": method, **stats})
    return pd.DataFrame(rows).sort_values("mean_mse").reset_index(drop=True)


def build_supplement_table_df(fold_mses: dict) -> pd.DataFrame:
    # Compact ACS rows for the env-augmented ERM robustness check.
    ordered_methods = ["EACS", "OLS", "Anchor", "Lasso", "Ridge-Int"]
    details = {
        "EACS": "soft gate",
        "OLS": "linear model on X",
        "Anchor": "state anchors",
        "Lasso": "lasso on X",
        "Ridge-Int": "bilinear ridge on X, u_e, and X ⊗ u_e",
    }
    rows = []
    for method in ordered_methods:
        if method not in fold_mses:
            continue
        stats = _finite_stats(fold_mses[method])
        if stats is None:
            continue
        rows.append({
            "dataset": "ACS income",
            "method": method,
            "detail": details.get(method, ""),
            **stats,
        })
    return pd.DataFrame(rows)


def _latex_escape_cell(x) -> str:
    text = str(x)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("⊗", r"$\otimes$")
    return text


def _write_latex_table(df: pd.DataFrame, path: str) -> None:
    cols = ["method", "detail", "mse_sd"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by income.py\n")
        f.write("\\begin{tabular}{lll}\n")
        f.write("\\toprule\n")
        f.write("Method & Detail & MSE (SD) " + r"\\" + "\n")
        f.write("\\midrule\n")
        for _, row in df[cols].iterrows():
            method = _latex_escape_cell(row["method"])
            detail = _latex_escape_cell(row["detail"])
            mse_sd = _latex_escape_cell(row["mse_sd"])
            f.write(f"{method} & {detail} & {mse_sd} " + r"\\" + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_summary_outputs(fold_mses: dict) -> None:
    full_df = build_summary_df(fold_mses)
    full_df.to_csv(ALL_RESULTS_CSV_PATH, index=False)
    print(f"➡️ Wrote full summary table to {ALL_RESULTS_CSV_PATH}")

    supp_df = build_supplement_table_df(fold_mses)
    supp_df.to_csv(SUPP_TABLE_CSV_PATH, index=False)
    _write_latex_table(supp_df, SUPP_TABLE_TEX_PATH)
    print(f"➡️ Wrote supplement table to {SUPP_TABLE_CSV_PATH}")
    print(f"➡️ Wrote supplement LaTeX table to {SUPP_TABLE_TEX_PATH}")
    print("\nSupplement table preview:")
    print(supp_df[["method", "detail", "mse_sd"]].to_string(index=False))


# Regenerate paper outputs from a cached results pickle without recomputing.
if os.environ.get("FORCE_RECOMPUTE") != "1" and os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH, "rb") as f:
        saved = pickle.load(f)
    fold_mses = saved.get("fold_mses")
    if fold_mses is not None and "Ridge-Int" in fold_mses:
        write_summary_outputs(fold_mses)
        main_methods = [m for m in ["EACS", "Anchor", "OLS", "Lasso"] if m in fold_mses]
        main_means = [float(np.mean(fold_mses[m])) for m in main_methods]
        main_stds = [float(np.std(fold_mses[m])) for m in main_methods]
        print_summary(main_methods, main_means, main_stds)
        plot_income_results(main_methods, main_means, main_stds)
        raise SystemExit

# Fresh compute
for i, te_days in enumerate(folds, start=1):
    tr = df[~df[ENV_COL].isin(te_days)].copy()
    te = df[df[ENV_COL].isin(te_days)].copy()
    tr_days = np.array([d for d in days if d not in te_days])

    # ── Outer-fold standardization (shared by ALL methods) ─────────────────
    fold_scaler = StandardScaler().fit(tr[feats].values.astype(float))
    tr_s, te_s = tr.copy(), te.copy()
    tr_s[feats] = tr_s[feats].astype(float)
    te_s[feats] = te_s[feats].astype(float)
    tr_s.loc[:, feats] = fold_scaler.transform(tr[feats].values.astype(float))
    te_s.loc[:, feats] = fold_scaler.transform(te[feats].values.astype(float))

    # Pre-compute inner splits on environments (reuse across methods)
    inner_days = tr_s[ENV_COL].unique()
    kf_inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    inner_splits_days = list(kf_inner.split(inner_days))

    # ── Anchor (tune γ on outer-train; evaluate on outer-test) ─────────────
    best_g, inner_best = tune_anchor_gamma(
        tr_s, inner_splits_days, gamma_grid, feats
    )
    outer_mse_anchor = float(anchor_fold_loss(tr_s, te_s, best_g, feats).mean())
    anchor_mses.append(outer_mse_anchor)
    anchor_gammas.append(best_g)

    # ── Lasso ──────────────────────────────

    alpha_grid = [1e-3, 1e-2, 1e-1, 1.0]
    alpha_mses = []
    for alpha in alpha_grid:
        mses = []
        for train_idx, val_idx in kf_inner.split(inner_days):
            dtr = inner_days[train_idx]
            dval = inner_days[val_idx]
            X_tr = tr_s.loc[tr_s[ENV_COL].isin(dtr), feats]
            y_tr = tr_s.loc[tr_s[ENV_COL].isin(dtr), TARGET_COL]
            X_val = tr_s.loc[tr_s[ENV_COL].isin(dval), feats]
            y_val = tr_s.loc[tr_s[ENV_COL].isin(dval), TARGET_COL]
            m = Lasso(alpha=alpha, max_iter=10000).fit(X_tr, y_tr)
            per_env = []
            for d in dval:
                env = tr_s[tr_s[ENV_COL] == d]
                per_env.append(
                    mean_squared_error(env[TARGET_COL], m.predict(env[feats]))
                )
            mses.append(float(np.mean(per_env)))
        alpha_mses.append((alpha, float(np.mean(mses))))
    best_alpha = min(alpha_mses, key=lambda x: x[1])[0]
    lasso_alphas.append(best_alpha)

    lasso_model = Lasso(alpha=best_alpha, max_iter=10000).fit(
        tr_s[feats], tr_s[TARGET_COL]
    )
    lasso_mses.append(
        np.mean([
            mean_squared_error(env[TARGET_COL], lasso_model.predict(env[feats]))
            for _, env in te_s.groupby(ENV_COL)
        ])
    )

    # ── OLS baseline ───────────────────────
    feat_scaler = fold_scaler
    X_cache_scaled_for_ols = {
        d: torch.tensor(feat_scaler.transform(X_cache_orig[d].cpu().numpy()),
                        dtype=torch.float32, device=DEVICE)
        for d in days
    }
    lr = LinearRegression().fit(
        np.vstack([X_cache_scaled_for_ols[d].cpu().numpy() for d in tr_days]),
        np.concatenate([y_cache[d].cpu().numpy() for d in tr_days]),
    )
    mse_ols = np.mean([
        mean_squared_error(
            y_cache[d].cpu().numpy(),
            lr.predict(X_cache_scaled_for_ols[d].cpu().numpy())
        )
        for d in te_days
    ])
    ols_mses.append(mse_ols)

    # ── EACS ───────────────────────────────
    ds_scaler = fold_scaler

    X_cache_scaled = {}
    stats_cache_unscaled = {}
    for d in days:
        X_scaled_np = ds_scaler.transform(X_cache_orig[d].cpu().numpy())
        X_cache_scaled[d] = torch.tensor(
            X_scaled_np, dtype=torch.float32, device=DEVICE
        )
        stats_cache_unscaled[d] = env_summary_stats_from_X(
            X_scaled_np, alpha_max=0.3
        )

    # Standardize summary coordinates across outer-training environments
    u_scaler = StandardScaler().fit(
        np.vstack([stats_cache_unscaled[d] for d in tr_days]).astype(float)
    )
    stats_cache_scaled = {
        d: torch.tensor(
            u_scaler.transform(stats_cache_unscaled[d].reshape(1, -1)).squeeze(0),
            dtype=torch.float32,
            device=DEVICE,
        )
        for d in days
    }

    bias_init = float(np.mean([y_cache[d].cpu().numpy().mean() for d in tr_days]))

    # ── Environment-augmented linear interaction baseline ─────────────────
    # Uses the same fixed summary u_e as the summary-MLP EACS gate, but learns
    # a direct predictor f(X_i,e, u_e) = b + γ^T u_e + X_i,e^T(β + B u_e).
    ridge_int_model, ridge_int_l2, ridge_int_inner = tune_ridge_interaction_summary(
        tr_days,
        inner_splits_days,
        X_cache_scaled,
        y_cache,
        stats_cache_scaled,
        bias_init=bias_init,
    )
    ridge_int_outer = _eval_ridge_interaction_model(
        ridge_int_model, te_days, X_cache_scaled, y_cache, stats_cache_scaled
    )
    ridge_int_l2s.append(ridge_int_l2)
    ridge_int_inner_mses.append(ridge_int_inner)
    ridge_int_mses.append(ridge_int_outer)

    # ── EACS helper routines ───────────────────────────────────────────────
    def _warmup_head(head, train_days, X_src, y_src, steps):
        for _ in range(steps):
            total_envs = len(train_days)
            loss_sum = 0.0
            head_opt = optim.Adam(head.parameters(), lr=HEAD_LR)
            head_opt.zero_grad()
            for d in train_days:
                X = X_src[d]
                y = y_src[d]
                z = torch.ones(1, p_feat, device=DEVICE)
                yhat = head((X * z).float()).squeeze(1)
                w = 1.0 / total_envs
                loss_sum = loss_sum + w * F.mse_loss(yhat, y)
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP_NORM)
            head_opt.step()

    def _train_joint(ds_embed, gate, head, temp,
                     train_days, X_src, y_src, use_ds):
        opt = optim.Adam(
            [{"params": (list(ds_embed.parameters()) if use_ds else []),
              "lr": GATE_LR, "weight_decay": 0.0},
             {"params": gate.parameters(),
              "lr": GATE_LR, "weight_decay": 0.0},
             {"params": head.parameters(),
              "lr": HEAD_LR, "weight_decay": 0.0}]
        )
        for _ in range(FINAL_EPOCHS - FINAL_WARMUP):
            opt.zero_grad()
            total_envs = len(train_days)
            loss_mse = 0.0
            for d in train_days:
                X = X_src[d]
                y = y_src[d]
                if use_ds:
                    e = ds_embed(X.unsqueeze(0))
                else:
                    e = stats_cache_scaled[d].unsqueeze(0)
                a = gate(e)
                z = soft_mask_from_logits(a, temp)
                yhat = head((X * z).float()).squeeze(1)
                w = 1.0 / total_envs
                loss_mse = loss_mse + w * F.mse_loss(yhat, y)
            loss_mse.backward()
            params = (list(ds_embed.parameters()) if use_ds else []) \
                     + list(gate.parameters()) + list(head.parameters())
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            opt.step()

    def _validate(ds_embed, gate, head, temp,
                  val_days, X_src, y_src, use_ds):
        with torch.no_grad():
            total_envs = len(val_days)
            se = 0.0
            for d in val_days:
                Xv = X_src[d]
                yv = y_src[d]
                if use_ds:
                    ev = ds_embed(Xv.unsqueeze(0))
                else:
                    ev = stats_cache_scaled[d].unsqueeze(0)
                av = gate(ev)
                zv = soft_mask_from_logits(av, temp)
                yhat = head((Xv * zv).float()).squeeze(1)
                w = 1.0 / total_envs
                se += w * F.mse_loss(yhat, yv).item()
            return float(se)

    # ── Inner CV: EACS‑DS ───────────────────────────────────────────────────
    best_ds_score, best_ds_cfg = float("inf"), None
    kf_inner_days = KFold(n_splits=INNER_FOLDS,
                          shuffle=True, random_state=SEED)

    for lam in LAMS:
        val_scores = []
        for tr_idx, val_idx in kf_inner_days.split(tr_days):
            dtr  = [tr_days[j] for j in tr_idx]
            dval = [tr_days[j] for j in val_idx]

            ds_embed = EnvDeepSetEmbed(p_feat, phi_hidden=HID,
                                       rho_hidden=HID, embed_dim=64).to(DEVICE)
            gate = GateMLPLogits(64, p_feat, hidden=HID).to(DEVICE)
            head = nn.Linear(p_feat, 1).to(DEVICE)
            with torch.no_grad():
                head.bias.data.fill_(bias_init)
                head.weight.data.zero_()

            # warm-up head
            _warmup_head(head, dtr, X_cache_scaled, y_cache, HYPER_WARMUP)

            # joint training for HP selection
            _train_joint(ds_embed, gate, head, TEMP,
                         dtr, X_cache_scaled, y_cache, use_ds=True)

            val_score = _validate(ds_embed, gate, head, TEMP,
                                  dval, X_cache_scaled, y_cache,
                                  use_ds=True)
            val_scores.append(val_score)

        avg_val = float(np.mean(val_scores))
        if avg_val < best_ds_score:
            best_ds_score = avg_val
            best_ds_cfg = {"lam": lam}

    # ── Inner CV: EACS‑NN (summary gate) ─────────────────────────────────────
    best_nn_score, best_nn_cfg = float("inf"), None
    for lam in LAMS:
        val_scores = []
        for tr_idx, val_idx in kf_inner_days.split(tr_days):
            dtr  = [tr_days[j] for j in tr_idx]
            dval = [tr_days[j] for j in val_idx]

            gate = GateMLPLogits(dim_u, p_feat, hidden=HID).to(DEVICE)
            head = nn.Linear(p_feat, 1).to(DEVICE)
            with torch.no_grad():
                head.bias.data.fill_(bias_init)
                head.weight.data.zero_()

            _warmup_head(head, dtr, X_cache_scaled, y_cache, HYPER_WARMUP)
            _train_joint(None, gate, head, TEMP,
                         dtr, X_cache_scaled, y_cache, use_ds=False)
            val_score = _validate(None, gate, head, TEMP,
                                  dval, X_cache_scaled, y_cache,
                                  use_ds=False)
            val_scores.append(val_score)

        avg_val = float(np.mean(val_scores))
        if avg_val < best_nn_score:
            best_nn_score = avg_val
            best_nn_cfg = {"lam": lam}

    # Decide best family
    if best_ds_score <= best_nn_score:
        chosen_family = "DS"
        chosen_cfg = best_ds_cfg

        ds_embed = EnvDeepSetEmbed(p_feat, phi_hidden=HID,
                                   rho_hidden=HID, embed_dim=64).to(DEVICE)
        gate_model = GateMLPLogits(64, p_feat, hidden=HID).to(DEVICE)
        head_model = nn.Linear(p_feat, 1).to(DEVICE)
        with torch.no_grad():
            head_model.bias.data.fill_(bias_init)
            head_model.weight.data.zero_()

        _warmup_head(head_model, tr_days, X_cache_scaled, y_cache,
                     FINAL_WARMUP)
        _train_joint(ds_embed, gate_model, head_model, TEMP,
                     tr_days, X_cache_scaled, y_cache, use_ds=True)
    else:
        chosen_family = "NN"
        chosen_cfg = best_nn_cfg

        gate_model = GateMLPLogits(dim_u, p_feat, hidden=HID).to(DEVICE)
        head_model = nn.Linear(p_feat, 1).to(DEVICE)
        with torch.no_grad():
            head_model.bias.data.fill_(bias_init)
            head_model.weight.data.zero_()

        _warmup_head(head_model, tr_days, X_cache_scaled, y_cache,
                     FINAL_WARMUP)
        _train_joint(None, gate_model, head_model, TEMP,
                     tr_days, X_cache_scaled, y_cache, use_ds=False)

    eacs_cfgs.append(chosen_family)

    # Evaluate EACS on outer test environments
    head_model.eval(); gate_model.eval()
    if chosen_family == "DS":
        ds_embed.eval()
    with torch.no_grad():
        total_envs = len(te_days)
        se = 0.0
        for d in te_days:
            Xv = X_cache_scaled[d]
            yv = y_cache[d]
            if chosen_family == "DS":
                ev = ds_embed(Xv.unsqueeze(0))
            else:
                ev = stats_cache_scaled[d].unsqueeze(0)
            av = gate_model(ev)
            zv = soft_mask_from_logits(av, TEMP)
            yhat = head_model((Xv * zv).float()).squeeze(1)
            w = 1.0 / total_envs
            se += w * F.mse_loss(yhat, yv).item()
        eacs_mses.append(float(se))

    print(f"\n--- Fold {i} summary ---")
    print(f"Anchor: γ={best_g:.4f}, outer‑CV MSE={outer_mse_anchor:.4f}, "
          f"inner‑CV MSE={inner_best:.4f}")
    print(f"Lasso:  α={best_alpha}, outer‑CV MSE={lasso_mses[-1]:.4f}")
    print(f"OLS:    outer‑CV MSE={mse_ols:.4f}")
    print(f"Ridge-Int: λ={ridge_int_l2:.4g}, outer‑CV MSE={ridge_int_outer:.4f}, "
          f"inner‑CV MSE={ridge_int_inner:.4f}")
    print(f"EACS:    family={chosen_family}, outer‑CV MSE={eacs_mses[-1]:.4f}")

# ─── Final summary / plot ───────────────────────────────────────────────────
fold_mses = {
    "EACS": eacs_mses,
    "Lasso": lasso_mses,
    "Ridge-Int": ridge_int_mses,
    "OLS": ols_mses,
    "Anchor": anchor_mses,
}

main_methods = ["EACS", "Lasso", "OLS", "Anchor"]
main_means = [float(np.mean(fold_mses[m])) for m in main_methods]
main_stds = [float(np.std(fold_mses[m])) for m in main_methods]

order = np.argsort(main_means)
main_methods = [main_methods[i] for i in order]
main_means   = [main_means[i]   for i in order]
main_stds    = [main_stds[i]    for i in order]

print_summary(main_methods, main_means, main_stds)

with open(PICKLE_PATH, "wb") as f:
    pickle.dump({
        "methods": main_methods,
        "means": main_means,
        "stds": main_stds,
        "fold_mses": fold_mses,
        "ridge_int_l2s": ridge_int_l2s,
        "ridge_int_inner_mses": ridge_int_inner_mses,
        "eacs_cfgs": eacs_cfgs,
    }, f)
print(f"➡️ Computed and saved metrics to {PICKLE_PATH}")

write_summary_outputs(fold_mses)

plot_income_results(main_methods, main_means, main_stds)
