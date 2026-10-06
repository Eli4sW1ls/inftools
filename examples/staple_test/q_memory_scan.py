"""
Scan all q[i, k] for start-interface dependence (memory) and test which CV explains it.

forward  (i < k-1): q[i, k] = P(reach lambda_k     | reached lambda_{k-1}, started at i)
backward (i > k+1): q[i, k] = P(reach lambda_k     | reached lambda_{k+1}, started at i)

Both are computed with the notebook (memory_analysis) formula; the trivial adjacent
start (i = k-1 forward, i = k+1 backward, always q = 1) is left out.

Stage 1 (path table only): for every (direction, k) and every start interface i with
an effective number of paths >= --min-neff, q_i and a heterogeneity statistic

    chi2 = sum_i neff_i (q_i - q_bar)^2 / (q_bar (1 - q_bar)),   df = (#i) - 1

with a permutation p-value (start labels shuffled over paths). Paths are treated as
independent; MC correlation makes the real p-values larger, hence the strict --alpha.

Stage 2 (order.txt): for the (direction, k) with significant memory, CV features at
the conditioning crossing (see q_cv_discrepancy.py; backward paths are mirrored, so
vz and dop are positive when moving along the path direction). For every CV a
logistic model success ~ CV + CV^2 is fitted on all participating paths, giving the
q per start interface that this CV alone predicts, q_pred_i, and

    chi2_res          = sum_i neff_i (q_i - q_pred_i)^2 / (q_bar (1 - q_bar))
    explained_excess  = (chi2 - chi2_res) / (chi2 - df)   (1 = all memory explained)
    p_res             = P(chi2_df > chi2_res)             (large = no memory left)

t_from_start_intf is reported but marked as a start proxy (it mostly encodes i), and
is left out of the all-CV model.

Usage:
    python q_memory_scan.py [--data-dir DIR] [--nskip 0] [--min-neff 20] [--alpha 1e-3]
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2 as chi2_dist

from q_cv_discrepancy import (DEFAULT_DATA_DIR, fit_logistic, cv_auc, features_for, kish_neff,
                              load_path_table, participants, read_order, style_axis)

START_PROXY = "t_from_start_intf"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#e5e5e2", "#eb6834"])


# --------------------------------------------------------------------------- #
# Stage 1: q per start interface and heterogeneity
# --------------------------------------------------------------------------- #
def group_stats(labels, W, y, n_groups):
    sw = np.bincount(labels, W, n_groups)
    sw2 = np.bincount(labels, W ** 2, n_groups)
    swy = np.bincount(labels, W * y, n_groups)
    return swy / sw, sw ** 2 / sw2


def chi2_stat(q, neff, q_bar):
    return np.sum(neff * (q - q_bar) ** 2) / (q_bar * (1 - q_bar))


def heterogeneity(part, min_neff, nperm, rng):
    neff_i = part.groupby("start")["W"].apply(lambda w: kish_neff(w.values))
    keep = neff_i.index[neff_i >= min_neff].values
    part = part[part["start"].isin(keep)].reset_index(drop=True)
    if len(keep) < 2:
        return None, part
    lab = np.searchsorted(keep, part["start"].values)
    W, y = part["W"].values, part["success"].values.astype(float)
    q_bar = np.sum(W * y) / np.sum(W)
    if q_bar in (0.0, 1.0):
        return None, part
    q, neff = group_stats(lab, W, y, len(keep))
    stat = chi2_stat(q, neff, q_bar)
    perm = np.empty(nperm)
    for b in range(nperm):
        qp, ne = group_stats(rng.permutation(lab), W, y, len(keep))
        perm[b] = chi2_stat(qp, ne, q_bar)
    res = {"starts": keep, "q": q, "neff": neff, "q_bar": q_bar, "chi2": stat,
           "df": len(keep) - 1, "p_perm": (np.sum(perm >= stat) + 1) / (nperm + 1),
           "p_chi2": chi2_dist.sf(stat, len(keep) - 1), "spread": q.max() - q.min()}
    return res, part


# --------------------------------------------------------------------------- #
# Stage 2: features and explanation
# --------------------------------------------------------------------------- #
def compute_features(configs, data_dir, interfaces, window, frame_dt, workers=16):
    """One read of order.txt per path, features for every config the path is in."""
    jobs = {}
    for key, part in configs.items():
        for idx, (pnr, start) in enumerate(part[["pnr", "start"]].values):
            jobs.setdefault(int(pnr), []).append((key, idx, int(start)))
    print(f"Reading {len(jobs)} order.txt files for {len(configs)} (direction, k) ...")

    def work(item):
        pnr, uses = item
        try:
            order = read_order(data_dir / "load" / str(pnr) / "order.txt")
        except Exception:
            return []
        return [(key, idx, features_for(order, key[0], start, key[1], interfaces, window, frame_dt))
                for key, idx, start in uses]

    out = {key: [None] * len(part) for key, part in configs.items()}
    with ThreadPoolExecutor(workers) as ex:
        for done, res in enumerate(ex.map(work, jobs.items()), 1):
            for key, idx, f in res:
                out[key][idx] = f
            if done % 5000 == 0:
                print(f"  {done}/{len(jobs)}")
    feats = {}
    for key, part in configs.items():
        ok = [f is not None for f in out[key]]
        rows = pd.DataFrame([f for f in out[key] if f is not None])
        feats[key] = pd.concat([part[ok].reset_index(drop=True), rows], axis=1)
    return feats


def explain(feat, het, feature_names):
    """Per CV: q_pred per start interface, explained excess heterogeneity, residual p."""
    starts, q_obs, neff, q_bar, df = het["starts"], het["q"], het["neff"], het["q_bar"], het["df"]
    lab = np.searchsorted(starts, feat["start"].values)
    W, y = feat["W"].values, feat["success"].values
    chi2_obs = chi2_stat(q_obs, neff, q_bar)

    def score(X, quadratic):
        predict = fit_logistic(X, y, W, quadratic)
        q_pred = np.bincount(lab, W * predict(X), len(starts)) / np.bincount(lab, W, len(starts))
        chi2_res = np.sum(neff * (q_obs - q_pred) ** 2) / (q_bar * (1 - q_bar))
        return {"explained_excess": (chi2_obs - chi2_res) / (chi2_obs - df) if chi2_obs > df else np.nan,
                "p_res": chi2_dist.sf(chi2_res, df), "chi2_res": chi2_res,
                "cvAUC": cv_auc(X, y, W, quadratic), "q_pred": q_pred}

    rows = []
    for f in feature_names:
        r = score(feat[[f]].values.astype(float), True)
        rows.append({"feature": f + (" (start proxy)" if f == START_PROXY else ""), **r})
    physical = [f for f in feature_names if f != START_PROXY]
    rows.append({"feature": "ALL CVs (linear)", **score(feat[physical].values.astype(float), False)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_scan(scan, interfaces, path):
    n = len(interfaces)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, direction in zip(axes, ("fw", "bw")):
        dev = np.full((n, n), np.nan)
        for _, r in scan[scan["dir"] == direction].iterrows():
            for i, q in zip(r["starts"], r["q"]):
                dev[r["k"], i] = q - r["q_bar"]
        im = ax.imshow(dev, cmap=DIVERGING, vmin=-0.3, vmax=0.3, origin="lower", aspect="auto")
        for _, r in scan[(scan["dir"] == direction) & scan["significant"]].iterrows():
            ax.text(n - 0.4, r["k"], "*", va="center", ha="left", fontsize=12, color="#0b0b0b")
        ax.set_xlabel("start interface i", fontsize=9)
        ax.set_ylabel("target interface k", fontsize=9)
        ax.set_title(("forward" if direction == "fw" else "backward") + ": q[i,k] − pooled q[·,k]",
                     fontsize=10)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlim(-0.5, n + 0.3)
        ax.tick_params(labelsize=7, colors="#52514e")
    fig.colorbar(im, ax=axes, shrink=0.8).ax.tick_params(labelsize=7)
    fig.text(0.01, 0.01, "cells: start interfaces with enough paths; * = significant start "
             "dependence", fontsize=8, color="#52514e")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_explained_heatmap(expl, path):
    keys = list(expl)
    feats = expl[keys[0]]["feature"].tolist()
    mat = np.array([expl[key].set_index("feature").loc[feats, "explained_excess"].values for key in keys])
    fig, ax = plt.subplots(figsize=(0.62 * len(feats) + 2.5, 0.45 * len(keys) + 2))
    im = ax.imshow(np.clip(mat, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if np.isfinite(mat[a, b]):
                v = mat[a, b]
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{d} k={k}" for d, k in keys], fontsize=8)
    ax.set_title("fraction of start-interface dependence of q explained by each CV", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_q_vs_start(expl, hets, path):
    keys = list(expl)
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.0 * nrow), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        het, e = hets[key], expl[key]
        x = het["starts"]
        err = np.sqrt(het["q"] * (1 - het["q"]) / het["neff"])
        ax.errorbar(x, het["q"], yerr=err, fmt="o", color="#0b0b0b", ms=5, capsize=2, label="observed")
        phys = e[~e["feature"].str.contains("proxy|ALL")]
        best = phys.loc[phys["explained_excess"].idxmax()]
        ax.plot(x, best["q_pred"], "-s", color="#2a78d6", ms=4, lw=1.8, label=f"{best['feature']}")
        allcv = e[e["feature"].str.startswith("ALL")].iloc[0]
        ax.plot(x, allcv["q_pred"], "--^", color="#eb6834", ms=4, lw=1.5, label="all CVs")
        ax.set_title(f"{key[0]} q[i,{key[1]}]  (p={het['p_perm']:.1g})", fontsize=9)
        ax.set_xlabel("start interface i", fontsize=8)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=6.5)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--nskip", type=int, default=0)
    ap.add_argument("--min-neff", type=float, default=20, help="min. effective paths per start interface")
    ap.add_argument("--alpha", type=float, default=1e-3, help="significance level for memory")
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--max-configs", type=int, default=16, help="max. (direction, k) analysed with CVs")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--frame-dt", type=float, default=0.2)
    ap.add_argument("--outdir", default="q_memory_scan")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df, weight, interfaces = load_path_table(data_dir, args.nskip)
    n = len(interfaces)
    print(f"{len(df)} paths after nskip={args.nskip}, {n} interfaces")

    # ---- stage 1 ---------------------------------------------------------- #
    rows, hets, parts = [], {}, {}
    for direction, ks in (("fw", range(2, n)), ("bw", range(0, n - 2))):
        for k in ks:
            het, part = heterogeneity(participants(df, weight, direction, k), args.min_neff,
                                      args.nperm, rng)
            if het is None:
                continue
            hets[(direction, k)], parts[(direction, k)] = het, part
            rows.append({"dir": direction, "k": k, **{c: het[c] for c in
                         ("q_bar", "chi2", "df", "p_perm", "p_chi2", "spread")},
                         "n_paths": len(part), "starts": het["starts"], "q": het["q"],
                         "neff": het["neff"]})
    scan = pd.DataFrame(rows)
    scan["significant"] = scan["p_perm"] < args.alpha
    scan["chi2/df"] = scan["chi2"] / scan["df"]
    fmt = lambda r: " ".join(f"{i}:{q:.2f}" for i, q in zip(r["starts"], r["q"]))
    scan["q_per_start"] = scan.apply(fmt, axis=1)
    pd.set_option("display.width", 250, "display.max_colwidth", 120, "display.precision", 3)
    print("\n=== Start-interface dependence of q[i,k] (q_per_start  i:q) ===")
    print(scan[["dir", "k", "n_paths", "q_bar", "chi2/df", "df", "p_perm", "spread", "significant",
                "q_per_start"]].to_string(index=False))
    scan.drop(columns=["starts", "q", "neff"]).to_csv(outdir / "memory_scan.csv", index=False)
    plot_scan(scan, interfaces, outdir / "memory_scan.png")

    # ---- stage 2 ---------------------------------------------------------- #
    sig = scan[scan["significant"]].sort_values("chi2/df", ascending=False).head(args.max_configs)
    if sig.empty:
        print("\nNo significant start dependence at this alpha.")
        return
    keys = [(r["dir"], r["k"]) for _, r in sig.iterrows()]
    feats = compute_features({key: parts[key] for key in keys}, data_dir, interfaces,
                             args.window, args.frame_dt)
    expl = {}
    for key in keys:
        feat = feats[key]
        names = [c for c in feat.columns if "@" in c or c.endswith("_win")
                 or c in ("t_since_prev_intf", START_PROXY)]
        good = feat[names].notna().all(1)
        feat = feat[good].reset_index(drop=True)
        het, _ = heterogeneity(feat[["start", "W", "success"]].assign(pnr=feat["pnr"]),
                               args.min_neff, 200, rng)
        if het is None:
            continue
        hets[key] = het
        e = explain(feat, het, names)
        expl[key] = e
        feat.to_csv(outdir / f"features_{key[0]}_k{key[1]}.csv", index=False)
        print(f"\n=== {key[0]} q[i,{key[1]}]: chi2/df = {het['chi2'] / het['df']:.2f} (df={het['df']}), "
              f"q per start: " + " ".join(f"{i}:{q:.2f}" for i, q in zip(het["starts"], het["q"])) + " ===")
        print(e.drop(columns=["q_pred"]).sort_values("explained_excess", ascending=False)
              .to_string(index=False))
    pd.concat([e.drop(columns=["q_pred"]).assign(dir=key[0], k=key[1]) for key, e in expl.items()]) \
        .to_csv(outdir / "memory_explained.csv", index=False)
    plot_explained_heatmap(expl, outdir / "memory_explained.png")
    plot_q_vs_start(expl, hets, outdir / "memory_q_vs_start.png")
    print(f"\nWrote tables and figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()
