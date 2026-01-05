"""Simulation A: sweep over selector feature representations"""

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
ALPHA_CI_BASELINE = 0.2
ALPHA_CI_SELECTOR = 0.1

SEED = 42
np.random.seed(SEED)

TRAIN_NOISE = 1.0

beta_coeff = {'y_c1': 1.0, 'y_c2': 1.0, 'x_c1': 1.0, 'x_c2': -1.0}
perturb_types = ['c1', 'c2', 'x']

n_train, n_test, n_sim = 100, 100, 1000
train_levels = np.linspace(0, 4, 100)
test_grid = np.linspace(0, 4, 20)

feature_sets = {
    'all': [0, 1, 2],
    'corr(c2,x)': [0],
    'std(c2)': [1],
    'std(x)': [2],
}
feature_names = list(feature_sets.keys())

PKL_PATH = 'sim_a_sta.pkl'
CSV_PATH = 'sim_a_sta_selector_acc.csv'


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


def sample_training_envs(noise_std):
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


def feature_vector_full(X_obs):
    c2, x = X_obs[:, 0], X_obs[:, 1]
    r = pearsonr(c2, x)[0]
    if np.isnan(r):
        r = 0.0
    return np.array([r, float(c2.std(ddof=1)), float(x.std(ddof=1))], dtype=float)


def _fit_classifier(feats, labels):
    if np.unique(labels).size < 2:
        clf = DummyClassifier(strategy='most_frequent')
        clf.fit(feats, labels)
        return clf

    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(feats, labels)
    return clf


def _training_features_and_labels(envs, models):
    m0, b2, b3, b23 = models

    feats = np.zeros((len(envs), 3), dtype=float)
    labels_nc = np.zeros(len(envs), dtype=int)
    labels_c = np.zeros(len(envs), dtype=int)

    p0 = predict_intercept(m0, n_train)

    for i, (_, _, X_full, Y) in enumerate(envs):
        Xo = X_full[:, 1:3]
        feats[i, :] = feature_vector_full(Xo)

        p_c2 = predict_c2(b2, Xo[:, 0])
        p_x = predict_x(b3, Xo[:, 1])
        p_c2x = predict_c2x(b23, Xo)

        mse_vals = [
            mse(Y, p0),
            mse(Y, p_c2),
            mse(Y, p_x),
            mse(Y, p_c2x),
        ]
        labels_nc[i] = int(np.argmin(mse_vals))
        labels_c[i] = int(np.argmin([mse_vals[1], mse_vals[3]]))

    return feats, labels_nc, labels_c


def run():
    base_c2 = {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types}
    base_c2x = {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types}

    eacs = {fs: {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types} for fs in feature_names}
    eacs_c = {fs: {pt: np.zeros((n_sim, len(test_grid))) for pt in perturb_types} for fs in feature_names}

    acc_eacs = {fs: np.zeros(n_sim) for fs in feature_names}
    acc_cau = {fs: np.zeros(n_sim) for fs in feature_names}

    for s in range(n_sim):
        X_pool, Y_pool, envs = sample_training_envs(noise_std=TRAIN_NOISE)
        models = fit_ols_models(X_pool, Y_pool)

        full_feats, labels_nc, labels_c = _training_features_and_labels(envs, models)

        selectors = {fs: _fit_classifier(full_feats[:, feature_sets[fs]], labels_nc) for fs in feature_names}
        selectors_c = {fs: _fit_classifier(full_feats[:, feature_sets[fs]], labels_c) for fs in feature_names}

        m0, b2, b3, b23 = models
        p0_test = predict_intercept(m0, n_test)

        correct_nc = {fs: 0 for fs in feature_names}
        correct_c = {fs: 0 for fs in feature_names}
        total = 0

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

                mse_all = [mse0, mse_c2, mse_x, mse_c2x]
                mse_cau = [mse_c2, mse_c2x]
                opt_nc = int(np.argmin(mse_all))
                opt_c = int(np.argmin(mse_cau))

                full_f = feature_vector_full(Xo)

                for fs in feature_names:
                    f = full_f[feature_sets[fs]].reshape(1, -1)

                    br = int(selectors[fs].predict(f)[0])
                    eacs[fs][pt][s, j] = mse_all[br]
                    if br == opt_nc:
                        correct_nc[fs] += 1

                    br_c = int(selectors_c[fs].predict(f)[0])
                    eacs_c[fs][pt][s, j] = mse_cau[br_c]
                    if br_c == opt_c:
                        correct_c[fs] += 1

                total += 1

        for fs in feature_names:
            acc_eacs[fs][s] = correct_nc[fs] / total
            acc_cau[fs][s] = correct_c[fs] / total

    mean_b2, se_b2 = {}, {}
    mean_b23, se_b23 = {}, {}
    for pt in perturb_types:
        mean_b2[pt], se_b2[pt] = mean_se(base_c2[pt])
        mean_b23[pt], se_b23[pt] = mean_se(base_c2x[pt])

    baseline = {'mean_b2': mean_b2, 'se_b2': se_b2, 'mean_b23': mean_b23, 'se_b23': se_b23}

    eacs_sum = {}
    for fs in feature_names:
        m_fs, se_fs = {}, {}
        for pt in perturb_types:
            m_fs[pt], se_fs[pt] = mean_se(eacs[fs][pt])
        eacs_sum[fs] = {'mean': m_fs, 'se': se_fs}

    eacs_c_sum = {}
    for fs in feature_names:
        m_fs, se_fs = {}, {}
        for pt in perturb_types:
            m_fs[pt], se_fs[pt] = mean_se(eacs_c[fs][pt])
        eacs_c_sum[fs] = {'mean': m_fs, 'se': se_fs}

    selector_acc = {}
    for fs in feature_names:
        m, se = mean_se(acc_eacs[fs], axis=0)
        mc, sec = mean_se(acc_cau[fs], axis=0)
        selector_acc[fs] = {'eacs': (float(m), float(se)), 'causal': (float(mc), float(sec))}

    return {'baseline': baseline, 'eacs': eacs_sum, 'eacs_constrained': eacs_c_sum, 'selector_acc': selector_acc}


def load_or_compute():
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and 'results' in obj:
            obj = obj['results']
        if isinstance(obj, dict) and TRAIN_NOISE in obj and isinstance(obj[TRAIN_NOISE], dict) and 'selector_acc' in obj[TRAIN_NOISE]:
            return obj[TRAIN_NOISE]
        if isinstance(obj, dict) and obj and 'selector_acc' in obj:
            return obj

    results = run()
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(results, f)
    return results


results = load_or_compute()


def _fmt(v):
    return f"{float(v):.3f}"


with open(CSV_PATH, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['representation', 'u', 'u_se', 'c', 'c_se'])
    for fs in feature_names:
        u_mean, u_se = results['selector_acc'][fs]['eacs']
        c_mean, c_se = results['selector_acc'][fs]['causal']
        w.writerow([fs, _fmt(u_mean), _fmt(u_se), _fmt(c_mean), _fmt(c_se)])


res = results

for include_constrained in [False, True]:
    fig, axs = plt.subplots(1, len(perturb_types), figsize=(12, 4), constrained_layout=True)
    for j, pt in enumerate(perturb_types):
        ax = axs[j]
        if j == 0:
            ax.set_ylabel('MSE')
        ax.set_title(f'Perturb on {pt}')
        ax.set_xticks([0, 1, 2, 3, 4])

        base_c2 = res['baseline']['mean_b2'][pt]
        se_c2 = res['baseline']['se_b2'][pt]
        base_c2x = res['baseline']['mean_b23'][pt]
        se_c2x = res['baseline']['se_b23'][pt]

        lbl_c2 = 'c2' if j == 0 else '_nolegend_'
        lbl_c2x = '{c2,x}' if j == 0 else '_nolegend_'
        ax.plot(test_grid, base_c2, color=BASELINE_C2_COLOR, label=lbl_c2, linewidth=LW_BASELINE)
        ax.fill_between(test_grid, base_c2 - 1.96 * se_c2, base_c2 + 1.96 * se_c2, color=BASELINE_C2_COLOR, alpha=ALPHA_CI_BASELINE)
        ax.plot(test_grid, base_c2x, color=BASELINE_C2X_COLOR, label=lbl_c2x, linewidth=LW_BASELINE)
        ax.fill_between(test_grid, base_c2x - 1.96 * se_c2x, base_c2x + 1.96 * se_c2x, color=BASELINE_C2X_COLOR, alpha=ALPHA_CI_BASELINE)

        for k, fs in enumerate(feature_names):
            col = plt.cm.Greens(0.8 - 0.6 * k / (len(feature_names) - 1))
            m = res['eacs'][fs]['mean'][pt]
            se = res['eacs'][fs]['se'][pt]
            lbl = fs if j == 0 else '_nolegend_'
            ax.plot(test_grid, m, color=col, label=lbl)
            ax.fill_between(test_grid, m - 1.96 * se, m + 1.96 * se, color=col, alpha=ALPHA_CI_SELECTOR)

        if include_constrained:
            for k, fs in enumerate(feature_names):
                col = plt.cm.Blues(0.8 - 0.6 * k / (len(feature_names) - 1))
                m = res['eacs_constrained'][fs]['mean'][pt]
                se = res['eacs_constrained'][fs]['se'][pt]
                lbl = f'cau {fs}' if j == 0 else '_nolegend_'
                ax.plot(test_grid, m, color=col, label=lbl)
                ax.fill_between(test_grid, m - 1.96 * se, m + 1.96 * se, color=col, alpha=ALPHA_CI_SELECTOR)

    legend_fs = 'small' if not include_constrained else 'x-small'
    axs[0].legend(loc='upper left', fontsize=legend_fs, frameon=True)
    fig.supxlabel('Perturb Level')
    plt.show()


plt.figure(figsize=(10, 6))
x = np.arange(len(feature_names))

means_u = [res['selector_acc'][fs]['eacs'][0] for fs in feature_names]
ses_u = [res['selector_acc'][fs]['eacs'][1] for fs in feature_names]
means_c = [res['selector_acc'][fs]['causal'][0] for fs in feature_names]
ses_c = [res['selector_acc'][fs]['causal'][1] for fs in feature_names]

col_u = plt.cm.Greens(0.6)
col_c = plt.cm.Blues(0.6)

h1 = plt.plot(x, means_u, marker='o', color=col_u, linewidth=2)[0]
plt.fill_between(x, np.array(means_u) - 1.96 * np.array(ses_u), np.array(means_u) + 1.96 * np.array(ses_u), color=col_u, alpha=0.1)

h2 = plt.plot(x, means_c, marker='s', color=col_c, linewidth=2)[0]
plt.fill_between(x, np.array(means_c) - 1.96 * np.array(ses_c), np.array(means_c) + 1.96 * np.array(ses_c), color=col_c, alpha=0.1)

plt.xlabel('Representation')
plt.ylabel('P(Pick True Best)')
plt.xticks(x, feature_names)
plt.legend([h1, h2], ['EACS', 'constrained'], frameon=True, fontsize='medium')
plt.show()
