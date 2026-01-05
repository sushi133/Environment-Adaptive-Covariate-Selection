"""Simulation A (base): pooled OLS baselines and environment statistics."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

plt.rcParams.update({'font.size': 15})
BASELINE_C2_COLOR = '#fdae61'
BASELINE_C2X_COLOR = '#d7191c'
LW_BASELINE = 2
ALPHA_CI_BASELINE = 0.2

SEED = 42
np.random.seed(SEED)

beta_coeff = {'y_c1': 1.0, 'y_c2': 1.0, 'x_c1': 1.0, 'x_c2': -1.0}
perturb_types = ['c1', 'c2', 'x']

n_train, n_test, n_sim = 100, 100, 1000
perturb_levels = np.linspace(0, 4, 20)


def mse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean((y - yhat) ** 2))


def generate_env_data(ptype, plevel, n, noise_std=1.0):
    c1 = np.random.normal(0, 1, n)
    c2 = np.random.normal(0, 1, n)
    if ptype == 'c1':
        c1 = c1 + plevel
    elif ptype == 'c2':
        c2 = c2 + np.random.normal(0, plevel, n)

    y = beta_coeff['y_c1'] * c1 + beta_coeff['y_c2'] * c2 + np.random.normal(0, noise_std, n)

    x = beta_coeff['x_c1'] * c1 + beta_coeff['x_c2'] * c2 + np.random.normal(0, 1, n)
    if ptype == 'x':
        x = x + np.random.normal(0, plevel, n)

    return np.column_stack([c1, c2, x]), y


def collect_training_data(levels, noise_std=1.0):
    xs, ys = [], []
    for pt in perturb_types:
        for lvl in levels:
            X, Y = generate_env_data(pt, float(lvl), n_train, noise_std=noise_std)
            xs.append(X[:, 1:3])
            ys.append(Y)
    return np.vstack(xs), np.hstack(ys)


def fit_ols_models(X, Y):
    n = len(Y)
    m0 = float(np.mean(Y))

    X_c2 = np.column_stack([np.ones(n), X[:, 0]])
    b2 = np.linalg.lstsq(X_c2, Y, rcond=None)[0]

    X_x = np.column_stack([np.ones(n), X[:, 1]])
    b3 = np.linalg.lstsq(X_x, Y, rcond=None)[0]

    X_c2x = np.column_stack([np.ones(n), X])
    b23 = np.linalg.lstsq(X_c2x, Y, rcond=None)[0]

    return m0, b2, b3, b23


def predict_intercept(m0, n):
    return np.full(n, m0)


def predict_c2(b2, c2):
    return b2[0] + b2[1] * c2


def predict_x(b3, x):
    return b3[0] + b3[1] * x


def predict_c2x(b23, X):
    return b23[0] + X @ b23[1:]


models_list = ['intercept', 'c2', 'x', 'c2+x']
mse_results = {
    f'{model}_Perturb_{pt}_{i}': []
    for model in models_list
    for pt in perturb_types
    for i in range(len(perturb_levels))
}


for _ in range(n_sim):
    X_train, Y_train = collect_training_data(perturb_levels, noise_std=1.0)
    m0, b2, b3, b23 = fit_ols_models(X_train, Y_train)

    p0_test = predict_intercept(m0, n_test)

    for pt in perturb_types:
        for i, lvl in enumerate(perturb_levels):
            X_full, Y_env = generate_env_data(pt, float(lvl), n_test, noise_std=1.0)
            X_obs = X_full[:, 1:3]

            mse_results[f'intercept_Perturb_{pt}_{i}'].append(mse(Y_env, p0_test))
            mse_results[f'c2_Perturb_{pt}_{i}'].append(mse(Y_env, predict_c2(b2, X_obs[:, 0])))
            mse_results[f'x_Perturb_{pt}_{i}'].append(mse(Y_env, predict_x(b3, X_obs[:, 1])))
            mse_results[f'c2+x_Perturb_{pt}_{i}'].append(mse(Y_env, predict_c2x(b23, X_obs)))


mse_summary = {}
for key, values in mse_results.items():
    arr = np.asarray(values)
    mse_summary[key] = (float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(n_sim)))


def plot_mse_subset(models_to_plot):
    """Plot MSE curves"""
    model_style = {
        'c2': {'label': 'c2', 'color': BASELINE_C2_COLOR},
        'c2+x': {'label': '{c2,x}', 'color': BASELINE_C2X_COLOR},
        'intercept': {'label': 'intercept', 'color': '0.5', 'linestyle': '--'},
        'x': {'label': 'x', 'color': 'tab:blue'},
    }

    fig, axs = plt.subplots(1, len(perturb_types), figsize=(12, 4), constrained_layout=True)

    for j, pt in enumerate(perturb_types):
        ax = axs[j]
        ax.set_title(f'Perturb on {pt}')
        if j == 0:
            ax.set_ylabel('MSE')

        for model in models_to_plot:
            means = [mse_summary[f'{model}_Perturb_{pt}_{i}'][0] for i in range(len(perturb_levels))]
            ses = [mse_summary[f'{model}_Perturb_{pt}_{i}'][1] for i in range(len(perturb_levels))]
            st = model_style.get(model, {'label': model})
            color = st.get('color', None)
            ls = st.get('linestyle', '-')
            label = st.get('label', model)
            ax.plot(perturb_levels, means, label=label, color=color, linestyle=ls, linewidth=LW_BASELINE)
            ax.fill_between(
                perturb_levels,
                np.array(means) - 1.96 * np.array(ses),
                np.array(means) + 1.96 * np.array(ses),
                color=color,
                alpha=ALPHA_CI_BASELINE,
            )

        ax.set_xticks([0, 1, 2, 3, 4])
        if j == 0:
            ax.legend(loc='upper left', frameon=True)

    fig.supxlabel('Perturb Level')
    plt.show()


plot_mse_subset(['c2', 'c2+x'])
plot_mse_subset(['intercept', 'x', 'c2', 'c2+x'])


# --- Environment summary diagnostics ---
fig, axs = plt.subplots(1, len(perturb_types), figsize=(12, 4), constrained_layout=True)

for j, pt in enumerate(perturb_types):
    corr_vals = []
    std_c2_vals = []
    std_x_vals = []

    for lvl in perturb_levels:
        r_sims = np.empty(n_sim, dtype=float)
        s2_sims = np.empty(n_sim, dtype=float)
        sx_sims = np.empty(n_sim, dtype=float)

        for k in range(n_sim):
            X, _ = generate_env_data(pt, float(lvl), n_test, noise_std=1.0)
            Xo = X[:, 1:3]
            r_sims[k] = pearsonr(Xo[:, 0], Xo[:, 1])[0]
            s2_sims[k] = float(Xo[:, 0].std(ddof=1))
            sx_sims[k] = float(Xo[:, 1].std(ddof=1))

        corr_vals.append(float(r_sims.mean()))
        std_c2_vals.append(float(s2_sims.mean()))
        std_x_vals.append(float(sx_sims.mean()))

    ax = axs[j]
    ax.set_title(f'Perturb on {pt}')
    if j == 0:
        ax.set_ylabel('Correlation')
    ax.set_ylim(-1, 1)
    ax.set_xticks([0, 1, 2, 3, 4])

    l1 = ax.plot(perturb_levels, corr_vals, color='tab:green', linewidth=LW_BASELINE)[0]

    ax2 = ax.twinx()
    l2 = ax2.plot(perturb_levels, std_c2_vals, color='tab:blue', marker='o', markersize=4, linewidth=LW_BASELINE)[0]
    l3 = ax2.plot(perturb_levels, std_x_vals, color='tab:orange', marker='s', markersize=4, linestyle='--', linewidth=LW_BASELINE)[0]
    ax2.set_ylim(0, 6)

    if j == 0:
        ax.legend([l1, l2, l3], ['corr(c2,x)', 'std(c2)', 'std(x)'], loc='upper left', frameon=True)
    if j == len(perturb_types) - 1:
        ax2.set_ylabel('Standard Deviations')

fig.supxlabel('Perturb Level')
plt.show()
