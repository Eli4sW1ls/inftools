"""
Which CV explains the start-interface dependence of the forward q[i, k]?

q[i, k] (i < k) is the conditional probability that a forward path starting at
interface i, having reached lambda_{k-1}, also reaches lambda_k. It is computed in
the notebook (memory_analysis) as

    q[i, k] = sum_{pe=i+1}^{k} W_pe[i, k:] / sum_{pe=i+1}^{k} W_pe[i, k-1:]

This script
  1. rebuilds the per-path ensemble weights exactly like
     compute_weight_matrices_weights (notebook) and reproduces q[i, k],
  2. extracts the paths that enter q[i, k] for the start interfaces in each group
     (default groups: i in {0,1}, {2}, {3..10}), with per-path weight
     W_p = sum_{pe=i+1}^{k} weight_p[pe] and outcome success = (end_intf >= k),
  3. reads load/<pnr>/order.txt for those paths and computes CV features at the
     conditioning point (first crossing of lambda_{k-1}), averaged over a window
     before it, and a few history features,
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
     small groups.

Usage:
    python q_cv_discrepancy.py --data-dir <infstapletis dir> [--k 12]
        [--groups "0,1;2;3-10"] [--nskip 50000] [--window 25] [--outdir DIR]
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


def q_forward(df, weight, i, k):
    """q[i, k] for i < k as in memory_analysis (notebook)."""
    n = weight.shape[1]
    sel = (weight[:, 0] == 0) & (df["start"].values == i) & (df["end"].values >= k - 1)
    W = weight[sel][:, i + 1:min(k, n - 1) + 1].sum(1)
    den = W.sum()
    num = W[df["end"].values[sel] >= k].sum()
    return num / den if den > 0 else np.nan, sel, W


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
    """
    op = order[:, 0]
    below_start = np.nonzero(op < lam_start)[0]
    t0 = int(below_start[0]) if len(below_start) else 0
    above = t0 + np.nonzero(op[t0:] >= lam_cross)[0]
    if len(above) == 0 or above[0] == 0:
        return None
    c = int(above[0])
    lo = max(c - window, 0)
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
    feats["t_from_start_intf"] = (c - below_start[below_start < c][-1]) * frame_dt
    return feats


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
# Plots
# --------------------------------------------------------------------------- #
def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e5e5e2", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_feature_grid(feat, labels, features, ref, path, nbins=8):
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
    axes[1, 0].set_ylabel("P(reach $\\lambda_k$ | $\\lambda_{k-1}$)", fontsize=8)
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


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--k", type=int, default=12, help="target interface of q[:, k]")
    ap.add_argument("--groups", default="0,1;2;3-10", help="start-interface groups, ';'-separated")
    ap.add_argument("--ref", type=int, default=-1, help="index of the reference group (default: last)")
    ap.add_argument("--nskip", type=int, default=0)
    ap.add_argument("--window", type=int, default=25, help="frames before the crossing to average over")
    ap.add_argument("--frame-dt", type=float, default=0.2, help="time between order.txt frames (dt*subcycles)")
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    k = args.k
    groups = parse_groups(args.groups)
    labels = [group_label(g) for g in groups]
    ref = args.ref % len(groups)
    outdir = pathlib.Path(args.outdir or f"q_cv_discrepancy_k{k}")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    df, weight, interfaces = load_path_table(data_dir, args.nskip)
    print(f"{len(df)} paths after nskip={args.nskip}; lambda_{k-1}={interfaces[k-1]}, "
          f"lambda_{k}={interfaces[k]}")

    # ---- 1. reproduce q[i, k] and collect participating paths ------------- #
    print(f"\nq[i,{k}] per start interface (notebook formula):")
    print(f"{'i':>3} {'n_paths':>8} {'n_succ':>7} {'neff':>7} {'q':>8}")
    rows = []
    for g, members in enumerate(groups):
        for i in members:
            q, sel, W = q_forward(df, weight, i, k)
            sub = df[sel].copy()
            sub["W"] = W
            sub["group"] = g
            sub = sub[sub["W"] > 0]
            sub["success"] = (sub["end"] >= k).astype(int)
            print(f"{i:>3} {len(sub):>8} {sub['success'].sum():>7} {kish_neff(sub['W'].values):7.1f} {q:8.4f}")
            rows.append(sub)
    part = pd.concat(rows, ignore_index=True)
    if np.any(part["maxop"].values[part["success"].values == 1] < interfaces[k]):
        print("warning: some 'successful' paths have maxop < lambda_k")

    # ---- 2. order.txt and features ---------------------------------------- #
    cache_file = outdir / f"orders_cache_nskip{args.nskip}.npz"
    orders = load_orders(data_dir / "load", part["pnr"].tolist(), cache_file)
    feat_rows = []
    for _, row in part.iterrows():
        order = orders.get(row["pnr"])
        f = path_features(order, interfaces[row["start"]], interfaces[k - 1], interfaces[k - 2],
                          args.window, args.frame_dt) if order is not None else None
        if f is None:
            print(f"  skipping path {row['pnr']} (no order.txt or no lambda_{k-1} crossing after start turn)")
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
                      outdir / "cv_at_crossing.png")
    plot_feature_grid(feat, labels, [f for f in features if not f.endswith("@cross")], ref,
                      outdir / "cv_window_history.png")
    plot_explained(metrics, labels, ref, outdir / "frac_explained.png")
    print(f"\nWrote tables and figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()
