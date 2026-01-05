"""Simulation A: sweep over number of environments"""

import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

plt.rcParams.update({'font.size': 15})
BASELINE_C2_COLOR = '#fdae61'
BASELINE_C2X_COLOR = '#d7191c'
LW_BASELINE = 2
LW_SELECTOR = 1.2
ALPHA_CI_BASELINE = 0.2
ALPHA_CI_SELECTOR = 0.1

SEED = 42
np.random.seed(SEED)

beta_coeff = {'y_c1': 1.0, 'y_c2': 1.0, 'x_c1': 1.0, 'x_c2': -1.0}
perturb_types = ['c1', 'c2', 'x']

noise_levels = [1.0, 5.0, 10.0]
test_grid = np.linspace(0, 4, 20)

env_counts = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5]
selected_envs = [100, 20, 5]

n_train, n_test, n_sim = 100, 100, 1000

PKL_PATH = "sim_a_env.pkl"
CSV_PATH = "sim_a_env_selector_acc.csv"


def mse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean((y - yhat) ** 2))


def mean_se(x, axis=0):
    x = np.asarray(x)
    m = x.mean(axis=axis)
    se = x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])
    return m, se


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


def sample_training_envs(train_levels, noise_std):
    envs, xs, ys = [], [], []
    for pt in perturb_types:
        for lvl in train_levels:
            X, Y = generate_env_data(pt, float(lvl), n_train, noise_std=noise_std)
            envs.append((pt, float(lvl), X, Y))
            xs.append(X[:, 1:3])
            ys.append(Y)
    return np.vstack(xs), np.hstack(ys), envs


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


def feature_vector(X_obs):
    c2, x = X_obs[:, 0], X_obs[:, 1]
    r = pearsonr(c2, x)[0]
    if np.isnan(r):
        r = 0.0
    return np.array([r, float(c2.std(ddof=1)), float(x.std(ddof=1))], dtype=float)


def _fit_classifier(feats, labels):
    if np.unique(labels).size < 2:
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(feats, labels)
        return clf

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(feats, labels)
    return clf


def fit_eacs_selectors(envs, models):
    m0, b2, b3, b23 = models
    feats, labels_nc, labels_c = [], [], []

    n = len(envs[0][3])
    p0 = predict_intercept(m0, n)

    for _, _, X_full, Y in envs:
        Xo = X_full[:, 1:3]
        feats.append(feature_vector(Xo))

        p_c2 = predict_c2(b2, Xo[:, 0])
        p_x = predict_x(b3, Xo[:, 1])
        p_c2x = predict_c2x(b23, Xo)

        mse_vals = [
            mse(Y, p0),
            mse(Y, p_c2),
            mse(Y, p_x),
            mse(Y, p_c2x),
        ]
        labels_nc.append(int(np.argmin(mse_vals)))
        labels_c.append(int(np.argmin([mse_vals[1], mse_vals[3]])))

    feats = np.vstack(feats)
    labels_nc = np.asarray(labels_nc)
    labels_c = np.asarray(labels_c)

    return _fit_classifier(feats, labels_nc), _fit_classifier(feats, labels_c)


def _summarize_pt(curves):
    mean, se = {}, {}
    for pt in perturb_types:
        m, s = mean_se(curves[pt])
        mean[pt] = m
        se[pt] = s
    return mean, se


def run_condition(n_env, noise_std):
    train_levels = np.linspace(0, 4, int(n_env))

    base_c2 = {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types}
    base_c2x = {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types}
    eacs = {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types}
    eacs_c = {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types}

    acc_eacs = np.zeros(n_sim)
    acc_causal = np.zeros(n_sim)

    for s in range(n_sim):
        X_pool, Y_pool, envs = sample_training_envs(train_levels, noise_std=noise_std)
        models = fit_ols_models(X_pool, Y_pool)

        sel, sel_c = fit_eacs_selectors(envs, models)

        m0, b2, b3, b23 = models
        p0_test = predict_intercept(m0, n_test)

        correct_nc, correct_c, total = 0, 0, 0

        for pt in perturb_types:
            for j, lvl in enumerate(test_grid):
                X_full, Y = generate_env_data(pt, float(lvl), n_test, noise_std=1.0)
                Xo = X_full[:, 1:3]

                p_c2 = predict_c2(b2, Xo[:, 0])
                p_x = predict_x(b3, Xo[:, 1])
                p_c2x = predict_c2x(b23, Xo)

                mse0 = mse(Y, p0_test)
                mse_c2 = mse(Y, p_c2)
                mse_x = mse(Y, p_x)
                mse_c2x = mse(Y, p_c2x)

                base_c2[pt][s, j] = mse_c2
                base_c2x[pt][s, j] = mse_c2x

                f = feature_vector(Xo).reshape(1, -1)

                mse_all = [mse0, mse_c2, mse_x, mse_c2x]
                br = int(sel.predict(f)[0])
                eacs[pt][s, j] = mse_all[br]

                opt_nc = int(np.argmin(mse_all))
                if br == opt_nc:
                    correct_nc += 1

                mse_cau = [mse_c2, mse_c2x]
                br_c = int(sel_c.predict(f)[0])
                eacs_c[pt][s, j] = mse_cau[br_c]

                opt_c = int(np.argmin(mse_cau))
                if br_c == opt_c:
                    correct_c += 1

                total += 1

        acc_eacs[s] = correct_nc / total
        acc_causal[s] = correct_c / total

    mean_b2, se_b2 = _summarize_pt(base_c2)
    mean_b23, se_b23 = _summarize_pt(base_c2x)
    mean_eacs, se_eacs = _summarize_pt(eacs)
    mean_eacs_c, se_eacs_c = _summarize_pt(eacs_c)

    baseline = {'mean_b2': mean_b2, 'se_b2': se_b2, 'mean_b23': mean_b23, 'se_b23': se_b23}
    eacs_sum = {'mean': mean_eacs, 'se': se_eacs}
    eacs_c_sum = {'mean': mean_eacs_c, 'se': se_eacs_c}

    acc_mean, acc_se = mean_se(acc_eacs, axis=0)
    acc_c_mean, acc_c_se = mean_se(acc_causal, axis=0)

    return {
        'baseline': baseline,
        'eacs': eacs_sum,
        'eacs_constrained': eacs_c_sum,
        'selector_acc': {
            'eacs': (float(acc_mean), float(acc_se)),
            'causal': (float(acc_c_mean), float(acc_c_se)),
        },
    }


def load_or_compute():
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "results" in obj:
            return obj["results"]
        return obj

    results = {}
    for noise in noise_levels:
        res_noise = {'baseline': {}, 'eacs': {}, 'eacs_constrained': {}, 'selector_acc': {}}
        for n_env in env_counts:
            out = run_condition(n_env, noise_std=float(noise))
            res_noise['baseline'][n_env] = out['baseline']
            res_noise['eacs'][n_env] = out['eacs']
            res_noise['eacs_constrained'][n_env] = out['eacs_constrained']
            res_noise['selector_acc'][n_env] = out['selector_acc']
        results[float(noise)] = res_noise

    with open(PKL_PATH, "wb") as f:
        pickle.dump(results, f)

    return results


results = load_or_compute()


def _sigma_tag(s):
    s = float(s)
    return str(int(s)) if s.is_integer() else str(s).replace('.', 'p')


def _fmt(v):
    return f"{float(v):.3f}"


with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    header = ["n_env"]
    for s in noise_levels:
        t = _sigma_tag(s)
        header += [f"sigma{t}_u", f"sigma{t}_u_se", f"sigma{t}_c", f"sigma{t}_c_se"]
    w.writerow(header)

    for n_env in env_counts:
        row = [n_env]
        for s in noise_levels:
            u_mean, u_se = results[float(s)]['selector_acc'][n_env]['eacs']
            c_mean, c_se = results[float(s)]['selector_acc'][n_env]['causal']
            row += [_fmt(u_mean), _fmt(u_se), _fmt(c_mean), _fmt(c_se)]
        w.writerow(row)


baseline_ref = env_counts[0]

for include_constrained in [False, True]:
    fig, axs = plt.subplots(len(noise_levels), len(perturb_types), figsize=(11, 7.5), constrained_layout=True)

    for i, noise in enumerate(noise_levels):
        res = results[float(noise)]
        for j, pt in enumerate(perturb_types):
            ax = axs[i, j]
            if i == 0:
                ax.set_title(f'Perturb on {pt}')
            if j == 0:
                ax.set_ylabel(f'MSE (σ={noise})')

            base_c2 = res['baseline'][baseline_ref]['mean_b2'][pt]
            se_c2 = res['baseline'][baseline_ref]['se_b2'][pt]
            base_c2x = res['baseline'][baseline_ref]['mean_b23'][pt]
            se_c2x = res['baseline'][baseline_ref]['se_b23'][pt]

            lbl_c2 = 'c2' if (i == 0 and j == 0) else '_nolegend_'
            lbl_c2x = '{c2,x}' if (i == 0 and j == 0) else '_nolegend_'
            ax.plot(test_grid, base_c2, label=lbl_c2, color=BASELINE_C2_COLOR, linewidth=LW_BASELINE)
            ax.fill_between(test_grid, base_c2 - 1.96 * se_c2, base_c2 + 1.96 * se_c2, color=BASELINE_C2_COLOR, alpha=ALPHA_CI_BASELINE)
            ax.plot(test_grid, base_c2x, label=lbl_c2x, color=BASELINE_C2X_COLOR, linewidth=LW_BASELINE)
            ax.fill_between(test_grid, base_c2x - 1.96 * se_c2x, base_c2x + 1.96 * se_c2x, color=BASELINE_C2X_COLOR, alpha=ALPHA_CI_BASELINE)

            for k, n_env in enumerate(env_counts):
                cg = plt.cm.Greens(0.8 - 0.6 * k / len(env_counts))
                ma = res['eacs'][n_env]['mean'][pt]
                se = res['eacs'][n_env]['se'][pt]
                lbl = f'{n_env} env' if (i == 0 and j == 0 and n_env in selected_envs) else '_nolegend_'
                ax.plot(test_grid, ma, color=cg, label=lbl, linewidth=LW_SELECTOR)
                ax.fill_between(test_grid, ma - 1.96 * se, ma + 1.96 * se, color=cg, alpha=ALPHA_CI_SELECTOR)

            if include_constrained:
                for k, n_env in enumerate(env_counts):
                    cb = plt.cm.Blues(0.8 - 0.6 * k / len(env_counts))
                    mc = res['eacs_constrained'][n_env]['mean'][pt]
                    se = res['eacs_constrained'][n_env]['se'][pt]
                    lbl = f'cau {n_env} env' if (i == 0 and j == 0 and n_env in selected_envs) else '_nolegend_'
                    ax.plot(test_grid, mc, color=cb, label=lbl, linewidth=LW_SELECTOR)
                    ax.fill_between(test_grid, mc - 1.96 * se, mc + 1.96 * se, color=cb, alpha=ALPHA_CI_SELECTOR)

            ax.set_xticks([0, 1, 2, 3, 4])

    legend_fs = 'small' if not include_constrained else 'xx-small'
    axs[0, 0].legend(loc='upper left', fontsize=legend_fs, frameon=True)
    fig.supxlabel('Perturb Level')
    plt.show()


plt.figure(figsize=(10, 6))
x_vals = sorted(env_counts)

cmap_u = plt.cm.Greens(np.linspace(0.8, 0.2, len(noise_levels)))
cmap_c = plt.cm.Blues(np.linspace(0.8, 0.2, len(noise_levels)))

handles, labels = [], []
for idx, noise in enumerate(noise_levels):
    acc = results[float(noise)]['selector_acc']
    means = [acc[n]['eacs'][0] for n in x_vals]
    ses = [acc[n]['eacs'][1] for n in x_vals]
    h = plt.plot(x_vals, means, marker='o', color=cmap_u[idx], linewidth=2)[0]
    plt.fill_between(x_vals, np.array(means) - 1.96 * np.array(ses), np.array(means) + 1.96 * np.array(ses), color=cmap_u[idx], alpha=0.1)
    handles.append(h)
    labels.append(f'EACS (σ={noise:.1f})')

for idx, noise in enumerate(noise_levels):
    acc = results[float(noise)]['selector_acc']
    means = [acc[n]['causal'][0] for n in x_vals]
    ses = [acc[n]['causal'][1] for n in x_vals]
    h = plt.plot(x_vals, means, marker='s', color=cmap_c[idx], linewidth=2)[0]
    plt.fill_between(x_vals, np.array(means) - 1.96 * np.array(ses), np.array(means) + 1.96 * np.array(ses), color=cmap_c[idx], alpha=0.1)
    handles.append(h)
    labels.append(f'constrained (σ={noise:.1f})')

plt.xlabel('Number of Environments')
plt.ylabel('P(Pick True Best)')
plt.xticks(x_vals)
plt.legend(handles, labels, frameon=True, fontsize='medium', ncol=2)
plt.show()
