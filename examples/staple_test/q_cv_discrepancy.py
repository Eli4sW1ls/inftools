"""
Which CV explains the start-interface dependence of q[i, k]?

Forward (--direction fw, i < k-1): q[i, k] is the probability that a forward path
starting at interface i, having reached lambda_{k-1}, also reaches lambda_k:

    q[i, k] = sum_{pe=i+1}^{k} W_pe[i, k:] / sum_{pe=i+1}^{k} W_pe[i, k-1:]

Backward (--direction bw, i > k+1): the probability that a backward path starting at
i, having reached lambda_{k+1}, also reaches lambda_k:

    q[i, k] = sum_{pe=k+2}^{i+1} W_pe[i, :k+1] / sum_{pe=k+2}^{i+1} W_pe[i, :k+2]

Both as in the notebook (memory_analysis). For backward paths the CV features are
computed on the mirrored path (lambda -> -lambda), so "first crossing of the
conditioning interface" means the first crossing of lambda_{k+1} coming down, and
vz / dop are positive when moving along the path direction. The lambda profiles are
always shown on the real lambda axis with the raw CV values.

This script
  1. rebuilds the per-path ensemble weights exactly like
     compute_weight_matrices_weights (notebook) and reproduces q[i, k],
  2. extracts the paths that enter q[i, k] for the start interfaces in each group
     (default: the start two interfaces back, i = k-2 / k+2, vs all other starts),
     with per-path weight W_p = the weight summed over the ensembles of the formula
     above and outcome success = reached lambda_k,
  3. reads load/<pnr>/order.txt for those paths and computes CV features at the
     conditioning point (first crossing of lambda_{k-1} / lambda_{k+1}), averaged
     over a window before it, and a few history features,
  4. reports metrics per CV:
       - group distributions:   weighted mean, SMD and KS distance vs reference group
       - outcome relevance:     weighted AUC of the CV for success (all paths) and
                                cross-validated AUC of a logistic model in the
                                reference group
       - gap explained:         a logistic model success ~ CV trained on the
                                reference group predicts q for every group;
                                frac_explained = (q_pred_G - q_ref) / (q_obs_G - q_ref)
     plus a permutation test of whether each group's q differs from the pool at
     all, and a per-path table (CV percentile within the reference group) for the
     small groups,
  5. bins all frames of every path by lambda and shows, per group (and split by
     outcome), the path-averaged CV per lambda bin and the 2D histogram
     P(CV | lambda) (--profile-until-cross: only frames up to the conditioning point).

Usage:
    python q_cv_discrepancy.py --data-dir <infstapletis dir> [--direction fw] [--k 12]
        [--groups "0,1;2;3-10"] [--nskip 0] [--window 25] [--outdir DIR]
    python q_cv_discrepancy.py --direction bw --k 4 --groups "6;7-13"
"""

import argparse
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomli
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

DEFAULT_DATA_DIR = (
    "/run/user/1001/gvfs/sftp:host=172.18.15.42,user=elias/home/elias/data/"
    "CG_P2C6_StapleTIS/infstapletis"
)
# Columns of order.txt after the time column (see OP.py: COM_Distance.calculate)
CV_NAMES = ["op", "theta", "nwater", "vz", "x", "y", "box_x"]
# Group colors: first three slots of the validated categorical palette
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# --------------------------------------------------------------------------- #
# Path data and weights (same logic as the notebook)
# --------------------------------------------------------------------------- #
def load_path_table(data_dir, nskip):
    with open(data_dir / "infretis.toml", "rb") as f:
        toml = tomli.load(f)
    interfaces = np.array(toml["simulation"]["interfaces"], dtype=float)
    n = len(interfaces)

    data = np.loadtxt(data_dir / "infretis_data.txt", dtype=str,
                      usecols=np.arange(5 + 2 * n))[nskip:]
    ptype = data[:, 4]
    start = np.full(len(ptype), -1)
    end = np.zeros(len(ptype), dtype=int)
    for idx, pt in enumerate(ptype):
        m = re.match(r"^(\d+)([LR]M[LR])(\d+)$", pt)
        if m:
            start[idx], end[idx] = int(m.group(1)), int(m.group(3))
    data[data == "----"] = "0.0"
    path_f = data[:, 5:5 + n].astype(float)
    path_w = data[:, 5 + n:5 + 2 * n].astype(float)

    # weight_k = path_f / min(path_w, 1) * sum(path_f) / sum(path_f / path_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nan_to_num(path_f / np.minimum(path_w, 1.0))
        norm = np.nan_to_num(path_f.sum(0) / np.nan_to_num(path_f / path_w).sum(0))
    weight = ratio * norm

    df = pd.DataFrame({
        "pnr": data[:, 0].astype(int),
        "len": data[:, 1].astype(int),
        "maxop": data[:, 2].astype(float),
        "minop": data[:, 3].astype(float),
        "ptype": ptype,
        "start": np.minimum(start, n - 1),
        "end": np.minimum(end, n - 1),
    })
    return df, weight, interfaces


def cond_interface(direction, k):
    """Index of the interface q[., k] is conditioned on (lambda_{k-1} or lambda_{k+1})."""
    return k - 1 if direction == "fw" else k + 1


def participants(df, weight, direction, k):
    """Paths entering q[., k] (all non-trivial start interfaces), weight W and outcome.

    The trivial adjacent start (i = k-1 forward, i = k+1 backward, q = 1) is excluded.
    """
    n = weight.shape[1]
    start, end = df["start"].values, df["end"].values
    non0 = weight[:, 0] == 0
    if direction == "fw":
        sel = non0 & (start < k - 1) & (end >= k - 1)
        # sum over ensembles pe = i+1 .. k
        lo, hi = start[sel] + 1, np.full(sel.sum(), min(k, n - 1))
        success = end[sel] >= k
    else:
        sel = non0 & (start > k + 1) & (end <= k + 1)
        # sum over ensembles pe = k+2 .. i+1
        lo, hi = np.full(sel.sum(), k + 2), np.minimum(start[sel] + 1, n - 1)
        success = end[sel] <= k
    cw = np.concatenate([np.zeros((sel.sum(), 1)), np.cumsum(weight[sel], 1)], 1)
    rows = np.arange(sel.sum())
    W = np.where(hi >= lo, cw[rows, hi + 1] - cw[rows, lo], 0.0)
    part = df[sel].copy()
    part["W"], part["success"] = W, success.astype(int)
    return part[part["W"] > 0].reset_index(drop=True)


def parse_groups(text):
    groups = []
    for part in text.split(";"):
        members = []
        for tok in part.split(","):
            if "-" in tok:
                a, b = tok.split("-")
                members += list(range(int(a), int(b) + 1))
            else:
                members.append(int(tok))
        groups.append(members)
    return groups


def group_label(members):
    if len(members) > 2 and members == list(range(members[0], members[-1] + 1)):
        return f"i={members[0]}..{members[-1]}"
    return "i=" + ",".join(map(str, members))


# --------------------------------------------------------------------------- #
# order.txt loading and features
# --------------------------------------------------------------------------- #
def read_order(path):
    arr = pd.read_csv(path, sep=r"\s+", comment="#", header=None).values
    return arr[:, 1:].astype(np.float32)


def load_orders(load_dir, pnrs, cache_file, workers=16):
    cache = {}
    if cache_file.exists():
        with np.load(cache_file) as z:
            cache = {int(key): z[key] for key in z.files}
    missing = [p for p in pnrs if p not in cache]
    if missing:
        print(f"Reading {len(missing)} order.txt files ({len(cache)} cached) ...")

        def _read(p):
            try:
                return p, read_order(load_dir / str(p) / "order.txt")
            except Exception as exc:  # missing / unreadable path folder
                print(f"  warning: path {p}: {exc}")
                return p, None

        with ThreadPoolExecutor(workers) as ex:
            for done, (p, arr) in enumerate(ex.map(_read, missing), 1):
                if arr is not None:
                    cache[p] = arr
                if done % 500 == 0:
                    print(f"  {done}/{len(missing)}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, **{str(p): a for p, a in cache.items()})
    return cache


def path_features(order, lam_start, lam_cross, lam_prev, window, frame_dt):
    """CV features at the first crossing of lam_cross (frame c) after the start turn.

    A staple path starting at interface a begins with a turn that crosses below
    lambda_a; for a = k-2 that turn starts above lambda_{k-1}, so the conditioning
    crossing is searched only after the path has first been below lambda_a.
    The averaging window starts no earlier than the last frame below lambda_a, so it
    covers only the approach from the start interface and never the start turn (which
    would make the _win features a proxy for the start interface).
    """
    op = order[:, 0]
    below_start = np.nonzero(op < lam_start)[0]
    t0 = int(below_start[0]) if len(below_start) else 0
    above = t0 + np.nonzero(op[t0:] >= lam_cross)[0]
    if len(above) == 0 or above[0] == 0:
        return None
    c = int(above[0])
    last_below_start = below_start[below_start < c][-1]
    lo = max(c - window, int(last_below_start))
    feats = {"cross_frame": c}
    for col, name in enumerate(CV_NAMES[1:], start=1):
        if col >= order.shape[1]:
            break
        feats[f"{name}@cross"] = order[c, col]
        feats[f"{name}_win"] = order[lo:c + 1, col].mean()
    feats["dop@cross"] = (op[c] - op[c - 1]) / frame_dt
    feats["dop_win"] = (op[c] - op[lo]) / ((c - lo) * frame_dt) if c > lo else np.nan
    below_prev = np.nonzero(op[:c] < lam_prev)[0]
    # time since the path was last below lambda_{k-2}: how directly it arrives
    feats["t_since_prev_intf"] = (c - below_prev[-1]) * frame_dt if len(below_prev) else np.nan
    # travel time from the start turn (last time below lambda_a) to the crossing
    feats["t_from_start_intf"] = (c - last_below_start) * frame_dt
    return feats


def features_for(order, direction, start, k, interfaces, window, frame_dt):
    """path_features for a forward path, or for the mirrored backward path."""
    if direction == "fw":
        return path_features(order, interfaces[start], interfaces[k - 1], interfaces[k - 2],
                             window, frame_dt)
    mirrored = order.copy()
    mirrored[:, 0] *= -1   # lambda
    mirrored[:, 3] *= -1   # vz: positive = moving along the path direction
    prev = interfaces[k + 2] if k + 2 < len(interfaces) else np.inf
    return path_features(mirrored, -interfaces[start], -interfaces[k + 1], -prev, window, frame_dt)


# --------------------------------------------------------------------------- #
# Weighted statistics
# --------------------------------------------------------------------------- #
def wmean(x, w):
    return np.sum(w * x) / np.sum(w)


def wstd(x, w):
    return np.sqrt(np.sum(w * (x - wmean(x, w)) ** 2) / np.sum(w))


def kish_neff(w):
    return np.sum(w) ** 2 / np.sum(w ** 2) if np.sum(w) > 0 else 0.0


def wcdf(x, w, grid):
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / np.sum(w)
    idx = np.searchsorted(x[order], grid, side="right") - 1
    return np.where(idx >= 0, cw[np.clip(idx, 0, None)], 0.0)


def wks(x1, w1, x2, w2):
    grid = np.union1d(x1, x2)
    return np.max(np.abs(wcdf(x1, w1, grid) - wcdf(x2, w2, grid)))


def wpercentile_of(x_ref, w_ref, value):
    return 100 * (np.sum(w_ref[x_ref < value]) + 0.5 * np.sum(w_ref[x_ref == value])) / np.sum(w_ref)


def wauc(x, y, w):
    """Weighted ROC AUC = P(x_success > x_failure); 0.5 = no information."""
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    xs, ws = x[neg], w[neg]
    order = np.argsort(xs)
    xs, cw = xs[order], np.concatenate([[0], np.cumsum(ws[order])])
    lo = np.searchsorted(xs, x[pos], side="left")
    hi = np.searchsorted(xs, x[pos], side="right")
    below = cw[lo] + 0.5 * (cw[hi] - cw[lo])
    return np.sum(w[pos] * below) / (np.sum(w[pos]) * np.sum(ws))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def design(X, mu, sd, quadratic):
    Z = (X - mu) / sd
    return np.hstack([Z, Z ** 2]) if quadratic else Z


def fit_logistic(X, y, w, quadratic=True, C=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(design(X, mu, sd, quadratic), y, sample_weight=w / w.mean())
    return lambda Xn: model.predict_proba(design(Xn, mu, sd, quadratic))[:, 1]


def cv_auc(X, y, w, quadratic=True, folds=5, seed=0):
    if min(np.sum(y == 1), np.sum(y == 0)) < folds:
        return np.nan
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        pred[te] = fit_logistic(X[tr], y[tr], w[tr], quadratic)(X[te])
    return wauc(pred, y, w)


def permutation_pvalue(y, w, n_group, q_obs, n_perm, rng):
    """P(|q(random subset of size n_group) - q_pool| >= |q_obs - q_pool|)."""
    q_pool = wmean(y, w)
    q_perm = np.empty(n_perm)
    for b in range(n_perm):
        idx = rng.choice(len(y), n_group, replace=False)
        q_perm[b] = wmean(y[idx], w[idx])
    return np.mean(np.abs(q_perm - q_pool) >= abs(q_obs - q_pool) - 1e-12), q_perm


def bootstrap_q(y, w, n_boot, rng):
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    return np.sum(w[idx] * y[idx], 1) / np.sum(w[idx], 1)


# --------------------------------------------------------------------------- #
# CVs as a function of lambda over the full paths
# --------------------------------------------------------------------------- #
def profile_bin_edges(feat, orders, lambda_bin, ncvb=40):
    """lambda bins of width lambda_bin and per-CV bins, from a subsample of all frames."""
    sample = np.concatenate([orders[p][::10] for p in feat["pnr"].values])
    # full lambda range: the start turns of the rare low-i paths must not be cut off
    lo, hi = sample[:, 0].min(), sample[:, 0].max()
    lam_edges = np.arange(np.floor(lo / lambda_bin) * lambda_bin, hi + lambda_bin, lambda_bin)
    cv_edges = []
    for col in range(1, sample.shape[1]):
        v = sample[:, col]
        a, b = np.quantile(v, [0.005, 0.995])
        if np.allclose(v, np.round(v)):  # integer CV (n_water): one bin per value
            cv_edges.append(np.arange(np.round(a) - 0.5, np.round(b) + 1.5))
        else:
            cv_edges.append(np.linspace(a, b, ncvb + 1))
    return lam_edges, cv_edges


def lambda_profiles(feat, orders, lam_edges, cv_edges, n_groups, until_cross=False):
    """Accumulate per-lambda-bin CV statistics per group and outcome.

    Every path is first averaged within each lambda bin it visits, so a path counts
    once per bin with its weight W regardless of how long it dwells there. The 2D
    histograms H[cv][g, lambda_bin, cv_bin] distribute that weight W over the CV
    values the path has in the bin.
    """
    nl, ncv = len(lam_edges) - 1, len(cv_edges)
    shape = (n_groups, 2, ncv, nl)
    S, S2 = np.zeros(shape), np.zeros(shape)
    Wb, W2b, Nb = (np.zeros((n_groups, 2, nl)) for _ in range(3))
    H = [np.zeros((n_groups, nl, len(e) - 1)) for e in cv_edges]
    for pnr, W, g, o, c in feat[["pnr", "W", "group", "success", "cross_frame"]].values:
        order = orders[int(pnr)]
        if until_cross:
            order = order[:int(c) + 1]
        lb = np.digitize(order[:, 0], lam_edges) - 1
        ok = (lb >= 0) & (lb < nl)
        lb = lb[ok]
        cnt = np.bincount(lb, minlength=nl)
        vis = cnt > 0
        g, o = int(g), int(o)
        for col in range(ncv):
            v = order[ok, col + 1].astype(float)
            mean_p = np.bincount(lb, weights=v, minlength=nl)[vis] / cnt[vis]
            S[g, o, col, vis] += W * mean_p
            S2[g, o, col, vis] += W * mean_p ** 2
            ncb = len(cv_edges[col]) - 1
            cb = np.clip(np.digitize(v, cv_edges[col]) - 1, 0, ncb - 1)
            H[col][g] += np.bincount(lb * ncb + cb, weights=W / cnt[lb],
                                     minlength=nl * ncb).reshape(nl, ncb)
        Wb[g, o, vis] += W
        W2b[g, o, vis] += W ** 2
        Nb[g, o, vis] += 1
    return {"S": S, "S2": S2, "Wb": Wb, "W2b": W2b, "Nb": Nb, "H": H,
            "lam_edges": lam_edges, "cv_edges": cv_edges}


def profile_stats(prof, outcome=None):
    """Weighted mean, standard error and number of paths per (group, cv, lambda bin).

    outcome=None pools successes and failures, otherwise 0 (failure) / 1 (success).
    """
    sl = slice(None) if outcome is None else slice(outcome, outcome + 1)
    S, S2 = prof["S"][:, sl].sum(1), prof["S2"][:, sl].sum(1)
    Wb, W2b = prof["Wb"][:, sl].sum(1)[:, None], prof["W2b"][:, sl].sum(1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = S / Wb
        var = np.clip(S2 / Wb - mean ** 2, 0, None)
        se = np.sqrt(var / (Wb ** 2 / W2b))
    return mean, se, prof["Nb"][:, sl].sum(1)


def profiles_table(prof, labels):
    centers = 0.5 * (prof["lam_edges"][1:] + prof["lam_edges"][:-1])
    rows = []
    for outcome, name in ((None, "all"), (1, "success"), (0, "failure")):
        mean, se, nb = profile_stats(prof, outcome)
        for g, lab in enumerate(labels):
            for col, cv in enumerate(CV_NAMES[1:1 + mean.shape[1]]):
                for b in np.nonzero(nb[g] > 0)[0]:
                    rows.append({"group": lab, "outcome": name, "cv": cv, "lambda": centers[b],
                                 "mean": mean[g, col, b], "se": se[g, col, b], "n_paths": nb[g, b]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e5e5e2", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_feature_grid(feat, labels, features, ref, path, k, cond, nbins=8):
    """Row 1: weighted CV distribution per group. Row 2: q vs CV (quantile bins) per group.

    Groups with fewer than 20 paths are drawn as individual paths (rug / outcome dots).
    """
    ncol = len(features)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 5.4), squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    for c, f in enumerate(features):
        x_all, w_all = feat[f].values, feat["W"].values
        ok = np.isfinite(x_all)
        edges = np.unique(np.quantile(x_all[ok], np.linspace(0, 1, nbins + 1)))
        centers = 0.5 * (edges[1:] + edges[:-1])
        hist_edges = np.linspace(*np.quantile(x_all[ok], [0.005, 0.995]), 31)
        for g, lab in enumerate(labels):
            m = (feat["group"].values == g) & ok
            if m.sum() == 0:
                continue
            x, w, y = x_all[m], w_all[m], feat["success"].values[m]
            color = GROUP_COLORS[g]
            if g == ref or m.sum() >= 20:
                axes[0, c].hist(x, bins=hist_edges, weights=w / w.sum(), histtype="step",
                                lw=2, color=color, label=lab)
                b = np.clip(np.digitize(x, edges[1:-1]), 0, len(centers) - 1)
                qb = np.array([wmean(y[b == j], w[b == j]) if np.any(b == j) else np.nan
                               for j in range(len(centers))])
                axes[1, c].plot(centers, qb, "-o", lw=2, ms=5, color=color, label=lab)
            else:  # too few paths for a histogram: show the individual paths
                axes[0, c].plot(x, np.full(len(x), -0.02 * (g + 1)), "|", ms=12,
                                mew=2, color=color, label=lab)
                axes[1, c].scatter(x, y + 0.04 * (g - 1), s=30 + 120 * w / w_all[ok].max(),
                                   color=color, edgecolor="white", lw=1, zorder=3, label=lab)
        axes[0, c].set_title(f, fontsize=9, color="#0b0b0b")
        axes[1, c].set_ylim(-0.1, 1.1)
        axes[1, c].set_xlabel(f, fontsize=8, color="#52514e")
        style_axis(axes[0, c])
        style_axis(axes[1, c])
    axes[0, 0].set_ylabel("weighted fraction", fontsize=8)
    axes[1, 0].set_ylabel(f"P(reach $\\lambda_{{{k}}}$ | $\\lambda_{{{cond}}}$)", fontsize=8)
    handles, labs = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_explained(metrics, labels, ref, path):
    others = [g for g in range(len(labels)) if g != ref]
    feats = metrics["feature"].unique()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(feats) + 1.5))
    height = 0.8 / len(others)
    for n, g in enumerate(others):
        vals = metrics.set_index("feature").loc[feats, f"frac_explained[{labels[g]}]"].values
        ax.barh(np.arange(len(feats)) + n * height, np.clip(vals, -1.5, 1.5), height=height - 0.04,
                color=GROUP_COLORS[g], label=f"{labels[g]} vs {labels[ref]}")
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.axvline(1, color="#52514e", lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(feats)) + 0.4 - height / 2)
    ax.set_yticklabels(feats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of q gap explained (clipped to ±1.5; 1 = fully explained)", fontsize=8)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mark_interfaces(ax, interfaces, k, cond):
    for n, lam in enumerate(interfaces):
        strong = n in (cond, k)
        ax.axvline(lam, color="#52514e" if strong else "#d6d5d0", lw=1.0 if strong else 0.6,
                   ls="--" if strong else "-", zorder=0)


def plot_lambda_profiles(prof, labels, interfaces, k, cond, path, min_paths=3):
    """One panel per CV: weighted mean CV vs lambda per group (band = +-1 SE)."""
    mean, se, nb = profile_stats(prof)
    centers = 0.5 * (prof["lam_edges"][1:] + prof["lam_edges"][:-1])
    names = CV_NAMES[1:1 + mean.shape[1]]
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.2 * nrow), squeeze=False)
    for col, cv in enumerate(names):
        ax = axes.flat[col]
        mark_interfaces(ax, interfaces, k, cond)
        for g, lab in enumerate(labels):
            ok = nb[g] >= min_paths
            m, s = np.where(ok, mean[g, col], np.nan), np.where(ok, se[g, col], np.nan)
            ax.fill_between(centers, m - s, m + s, color=GROUP_COLORS[g], alpha=0.18, lw=0)
            ax.plot(centers, m, lw=2, color=GROUP_COLORS[g], label=lab)
        ax.set_title(cv, fontsize=9, color="#0b0b0b")
        ax.set_xlabel("$\\lambda$", fontsize=8, color="#52514e")
        style_axis(ax)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    handles, labs = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"path-averaged CV per $\\lambda$ bin (band ±1 SE, bins with ≥{min_paths} paths; "
                 f"dashed: $\\lambda_{{{cond}}}$, $\\lambda_{{{k}}}$)", y=0.99, fontsize=9, color="#52514e")
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lambda_profiles_outcome(prof, labels, interfaces, k, cond, path, min_paths=3):
    """Rows: CVs, columns: groups. Solid = reaches lambda_k, dashed = turns back.

    The y axis is shared along each row; its range follows the plotted means (not the
    SE bands, which blow up for the small groups).
    """
    stats = {o: profile_stats(prof, o) for o in (1, 0)}
    centers = 0.5 * (prof["lam_edges"][1:] + prof["lam_edges"][:-1])
    names = CV_NAMES[1:1 + stats[1][0].shape[1]]
    fig, axes = plt.subplots(len(names), len(labels), figsize=(4.6 * len(labels), 2.3 * len(names)),
                             squeeze=False, sharex=True, sharey="row")
    for col, cv in enumerate(names):
        row_means = []
        for g, lab in enumerate(labels):
            ax = axes[col, g]
            mark_interfaces(ax, interfaces, k, cond)
            for o, ls, name in ((1, "-", f"reaches $\\lambda_{{{k}}}$"), (0, "--", "turns back")):
                mean, se, nb = stats[o]
                ok = nb[g] >= min_paths
                m, s = np.where(ok, mean[g, col], np.nan), np.where(ok, se[g, col], np.nan)
                ax.fill_between(centers, m - s, m + s, color=GROUP_COLORS[g], alpha=0.15, lw=0)
                ax.plot(centers, m, ls=ls, lw=1.8, color=GROUP_COLORS[g], label=name)
                row_means.append(m)
            style_axis(ax)
            if col == 0:
                ax.set_title(lab, fontsize=10, color="#0b0b0b")
                ax.legend(frameon=False, fontsize=7, loc="best")
            if g == 0:
                ax.set_ylabel(cv, fontsize=9)
            if col == len(names) - 1:
                ax.set_xlabel("$\\lambda$", fontsize=8, color="#52514e")
        lo, hi = np.nanmin(row_means), np.nanmax(row_means)
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        axes[col, 0].set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_lambda_hist2d(prof, labels, interfaces, k, cond, path, min_paths=3):
    """Rows: CVs, columns: groups. Colour: P(CV | lambda) (each lambda column normalised);
    line: weighted mean."""
    mean, _, nb = profile_stats(prof)
    lam_edges = prof["lam_edges"]
    centers = 0.5 * (lam_edges[1:] + lam_edges[:-1])
    names = CV_NAMES[1:1 + len(prof["H"])]
    fig, axes = plt.subplots(len(names), len(labels), figsize=(4.6 * len(labels), 2.3 * len(names)),
                             squeeze=False, sharex=True)
    for col, cv in enumerate(names):
        edges = prof["cv_edges"][col]
        for g, lab in enumerate(labels):
            ax = axes[col, g]
            h = prof["H"][col][g]
            with np.errstate(invalid="ignore", divide="ignore"):
                h = h / h.sum(1, keepdims=True)
            h[nb[g] < min_paths] = np.nan
            mesh = ax.pcolormesh(lam_edges, edges, h.T, cmap="Blues", vmin=0,
                                 vmax=np.nanquantile(h, 0.99) if np.isfinite(h).any() else 1,
                                 shading="flat", rasterized=True)
            ax.plot(centers, np.where(nb[g] >= min_paths, mean[g, col], np.nan),
                    color="#0b0b0b", lw=1.2)
            for n in (cond, k):
                ax.axvline(interfaces[n], color="#eb6834", lw=1, ls="--")
            ax.tick_params(colors="#52514e", labelsize=7)
            if col == 0:
                ax.set_title(lab, fontsize=10, color="#0b0b0b")
            if g == 0:
                ax.set_ylabel(cv, fontsize=9)
            if g == len(labels) - 1:
                fig.colorbar(mesh, ax=ax, pad=0.01).ax.tick_params(labelsize=6)
            if col == len(names) - 1:
                ax.set_xlabel("$\\lambda$", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--direction", choices=("fw", "bw"), default="fw",
                    help="forward q (reach lambda_k from lambda_{k-1}) or backward (from lambda_{k+1})")
    ap.add_argument("--k", type=int, default=12, help="target interface of q[:, k]")
    ap.add_argument("--groups", default=None,
                    help="start-interface groups, ';'-separated, e.g. \"0,1;2;3-10\" "
                         "(default: i=k-2 vs i<k-2 forward, i=k+2 vs i>k+2 backward)")
    ap.add_argument("--ref", type=int, default=-1, help="index of the reference group (default: last)")
    ap.add_argument("--nskip", type=int, default=0)
    ap.add_argument("--window", type=int, default=25, help="frames before the crossing to average over")
    ap.add_argument("--frame-dt", type=float, default=0.2, help="time between order.txt frames (dt*subcycles)")
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--lambda-bin", type=float, default=0.5, help="lambda bin width of the CV profiles")
    ap.add_argument("--profile-until-cross", action="store_true",
                    help="CV profiles only up to the conditioning crossing instead of the full path")
    ap.add_argument("--min-paths", type=int, default=3, help="min. paths per lambda bin to draw a profile")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    k, direction = args.k, args.direction
    cond = cond_interface(direction, k)
    rng = np.random.default_rng(0)

    df, weight, interfaces = load_path_table(data_dir, args.nskip)
    n_int = len(interfaces)
    if not 0 <= cond < n_int or not 0 <= k < n_int:
        raise SystemExit(f"k={k} has no conditioning interface for direction {direction}")
    if args.groups is None:
        args.groups = f"{k - 2};0-{k - 3}" if direction == "fw" else f"{k + 2};{k + 3}-{n_int - 1}"
    groups = parse_groups(args.groups)
    labels = [group_label(g) for g in groups]
    ref = args.ref % len(groups)
    outdir = pathlib.Path(args.outdir or (f"q_cv_discrepancy_k{k}" if direction == "fw"
                                          else f"q_cv_discrepancy_bw_k{k}"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} paths after nskip={args.nskip}; {direction}: reach lambda_{k}={interfaces[k]} "
          f"given lambda_{cond}={interfaces[cond]}")

    # ---- 1. reproduce q[i, k] and collect participating paths ------------- #
    allpart = participants(df, weight, direction, k)
    print(f"\nq[i,{k}] per start interface (notebook formula):")
    print(f"{'i':>3} {'n_paths':>8} {'n_succ':>7} {'neff':>7} {'q':>8}")
    rows = []
    for g, members in enumerate(groups):
        for i in members:
            valid = i < k - 1 if direction == "fw" else i > k + 1
            if not valid:
                print(f"{i:>3}  skipped: not a non-trivial start interface for {direction} q[i,{k}]")
                continue
            sub = allpart[allpart["start"] == i].copy()
            sub["group"] = g
            q = wmean(sub["success"].values, sub["W"].values) if len(sub) else np.nan
            print(f"{i:>3} {len(sub):>8} {sub['success'].sum():>7} {kish_neff(sub['W'].values):7.1f} {q:8.4f}")
            rows.append(sub)
    part = pd.concat(rows, ignore_index=True)
    succ = part["success"].values == 1
    if direction == "fw" and np.any(part["maxop"].values[succ] < interfaces[k]):
        print("warning: some 'successful' paths have maxop < lambda_k")
    if direction == "bw" and np.any(part["minop"].values[succ] > interfaces[k]):
        print("warning: some 'successful' paths have minop > lambda_k")

    # ---- 2. order.txt and features ---------------------------------------- #
    cache_file = outdir / f"orders_cache_nskip{args.nskip}.npz"
    orders = load_orders(data_dir / "load", part["pnr"].tolist(), cache_file)
    feat_rows = []
    for _, row in part.iterrows():
        order = orders.get(row["pnr"])
        f = features_for(order, direction, int(row["start"]), k, interfaces, args.window,
                         args.frame_dt) if order is not None else None
        if f is None:
            print(f"  skipping path {row['pnr']} (no order.txt or no lambda_{cond} crossing after start turn)")
            continue
        feat_rows.append({**row.to_dict(), **f})
    feat = pd.DataFrame(feat_rows)
    features = [c for c in feat.columns if "@" in c or c.endswith("_win")
                or c in ("t_since_prev_intf", "t_from_start_intf")]
    feat.to_csv(outdir / "participating_paths_features.csv", index=False)

    y_all, w_all, g_all = feat["success"].values, feat["W"].values, feat["group"].values

    # ---- 3. is there a real difference? ------------------------------------ #
    print(f"\n=== Group q[., {k}] (reference: {labels[ref]}) ===")
    print(f"{'group':>12} {'n':>6} {'neff':>7} {'q':>7} {'95% boot CI':>17} {'perm p (vs pool)':>17}")
    q_obs, summary = {}, []
    for g, lab in enumerate(labels):
        m = g_all == g
        y, w = y_all[m], w_all[m]
        q_obs[g] = wmean(y, w)
        boot = bootstrap_q(y, w, args.nboot, rng)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p, _ = permutation_pvalue(y_all, w_all, m.sum(), q_obs[g], args.nperm, rng)
        print(f"{lab:>12} {m.sum():>6} {kish_neff(w):7.1f} {q_obs[g]:7.4f}   [{lo:.3f}, {hi:.3f}] {p:17.4f}")
        summary.append({"group": lab, "n": m.sum(), "neff": kish_neff(w), "q": q_obs[g],
                        "ci_lo": lo, "ci_hi": hi, "perm_p": p})
    pd.DataFrame(summary).to_csv(outdir / "group_q_summary.csv", index=False)
    print("(bootstrap/permutation treat paths as independent; MC correlation makes the "
          "true uncertainty larger)")

    # ---- 4. per-CV metrics ------------------------------------------------- #
    mref = g_all == ref
    metrics = []
    for f in features:
        x = feat[f].values.astype(float)
        ok = np.isfinite(x)
        rec = {"feature": f, "AUC_success_all": wauc(x[ok], y_all[ok], w_all[ok])}
        okr = ok & mref
        rec["cvAUC_ref_model"] = cv_auc(x[okr, None], y_all[okr], w_all[okr])
        predict = fit_logistic(x[okr, None], y_all[okr], w_all[okr])
        sd_ref = wstd(x[okr], w_all[okr])
        for g, lab in enumerate(labels):
            m = ok & (g_all == g)
            rec[f"mean[{lab}]"] = wmean(x[m], w_all[m])
            if g == ref:
                continue
            rec[f"SMD[{lab}]"] = (rec[f"mean[{lab}]"] - wmean(x[okr], w_all[okr])) / sd_ref
            rec[f"KS[{lab}]"] = wks(x[m], w_all[m], x[okr], w_all[okr])
            q_pred_g = wmean(predict(x[m, None]), w_all[m])
            q_pred_ref = wmean(predict(x[okr, None]), w_all[okr])
            rec[f"q_pred[{lab}]"] = q_pred_g
            gap = q_obs[g] - q_obs[ref]
            rec[f"frac_explained[{lab}]"] = (q_pred_g - q_pred_ref) / gap if abs(gap) > 1e-9 else np.nan
        metrics.append(rec)
    metrics = pd.DataFrame(metrics)

    # all CVs together
    good = [f for f in features if np.isfinite(feat[f].values).all()]
    X = feat[good].values.astype(float)
    predict_all = fit_logistic(X[mref], y_all[mref], w_all[mref], quadratic=False)
    print(f"\n=== Multivariate logistic model (trained on {labels[ref]}, {len(good)} CVs, linear) ===")
    print(f"cross-validated AUC in reference group: "
          f"{cv_auc(X[mref], y_all[mref], w_all[mref], quadratic=False):.3f}")
    for g, lab in enumerate(labels):
        m = g_all == g
        print(f"  {lab:>12}: q_obs = {q_obs[g]:.4f}   q_pred(all CVs) = "
              f"{wmean(predict_all(X[m]), w_all[m]):.4f}")

    pd.set_option("display.width", 250, "display.max_columns", 50, "display.precision", 3)
    others = [g for g in range(len(labels)) if g != ref]
    print("\n=== Per-CV metrics ===")
    print("AUC_success_all : weighted AUC of the CV for reaching lambda_k (0.5 = no info), all groups")
    print("cvAUC_ref_model : 5-fold CV AUC of logistic(CV, CV^2) inside the reference group")
    print("SMD / KS        : shift of the CV distribution of the group vs the reference group")
    print("frac_explained  : (q_pred_G - q_pred_ref)/(q_obs_G - q_obs_ref) from the reference-group model")
    for g in others:
        lab = labels[g]
        cols = ["feature", "AUC_success_all", "cvAUC_ref_model", f"mean[{lab}]", f"mean[{labels[ref]}]",
                f"SMD[{lab}]", f"KS[{lab}]", f"q_pred[{lab}]", f"frac_explained[{lab}]"]
        print(f"\n--- {lab} vs {labels[ref]}:  q_obs = {q_obs[g]:.4f} vs {q_obs[ref]:.4f} ---")
        print(metrics[cols].sort_values(f"frac_explained[{lab}]", key=lambda s: -s.abs().clip(upper=2))
              .to_string(index=False))
    metrics.to_csv(outdir / "cv_metrics.csv", index=False)

    # ---- 5. per-path table for the small groups ---------------------------- #
    key = ["theta@cross", "nwater_win", "vz@cross", "dop@cross", "box_x@cross", "t_since_prev_intf"]
    key = [f for f in key if f in features]
    for g in others:
        m = g_all == g
        if m.sum() > 30:
            continue
        print(f"\n=== Paths in {labels[g]}: value (percentile within {labels[ref]}) ===")
        tab = feat.loc[m, ["pnr", "ptype", "start", "end", "W", "success"]].copy()
        for f in key:
            xr, wr = feat.loc[mref, f].values, w_all[mref]
            xr, wr = xr[np.isfinite(xr)], wr[np.isfinite(xr)]
            tab[f] = [f"{v:8.3f} ({wpercentile_of(xr, wr, v):3.0f}%)" for v in feat.loc[m, f].values]
        print(tab.to_string(index=False))

    # ---- 6. figures -------------------------------------------------------- #
    plot_feature_grid(feat, labels, [f for f in features if f.endswith("@cross")], ref,
                      outdir / "cv_at_crossing.png", k, cond)
    plot_feature_grid(feat, labels, [f for f in features if not f.endswith("@cross")], ref,
                      outdir / "cv_window_history.png", k, cond)
    plot_explained(metrics, labels, ref, outdir / "frac_explained.png")

    # ---- 7. CVs vs lambda over the full paths ------------------------------ #
    lam_edges, cv_edges = profile_bin_edges(feat, orders, args.lambda_bin)
    prof = lambda_profiles(feat, orders, lam_edges, cv_edges, len(labels), args.profile_until_cross)
    tag = "_until_cross" if args.profile_until_cross else ""
    profiles_table(prof, labels).to_csv(outdir / f"cv_lambda_profiles{tag}.csv", index=False)
    plot_lambda_profiles(prof, labels, interfaces, k, cond, outdir / f"cv_vs_lambda{tag}.png",
                         args.min_paths)
    plot_lambda_profiles_outcome(prof, labels, interfaces, k, cond,
                                 outdir / f"cv_vs_lambda_outcome{tag}.png", args.min_paths)
    plot_lambda_hist2d(prof, labels, interfaces, k, cond, outdir / f"cv_vs_lambda_hist2d{tag}.png",
                       args.min_paths)
    print(f"\nWrote tables and figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()
