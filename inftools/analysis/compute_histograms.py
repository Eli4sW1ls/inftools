"""
Compute histograms and free energies for infretis simulation data.

This module uses the native inftools WHAM and Free_energy functions directly.

Output: CSV files with histogram data and free energies.
"""
import os
from pathlib import Path
from typing import Annotated, Optional, List, Dict, Tuple
import numpy as np
import typer

# Import inftools utilities
from inftools.misc.infinit_helper import read_toml
from inftools.misc.data_helper import data_reader
from inftools.analysis.toolsWHAM import get_WHAMfactors
from inftools.analysis.Free_energy import extract, update_histogram, calculate_free_energy


def compute_all_histograms(
    toml: Annotated[str, typer.Option("-toml", help="The .toml file")] = "infretis.toml",
    data: Annotated[str, typer.Option("-data", help="The infretis_data.txt file")] = "infretis_data.txt",
    trajdir: Annotated[str, typer.Option("-trajdir", help="Directory with path folders")] = "load",
    outdir: Annotated[str, typer.Option("-outdir", help="Output directory for CSV files")] = "histograms",
    nskip: Annotated[int, typer.Option("-nskip", help="Skip first N paths")] = 0,
    dlambda: Annotated[Optional[float], typer.Option("-dlambda", help="Bin width (mutually exclusive with -nbins)")] = None,
    nbins: Annotated[Optional[int], typer.Option("-nbins", help="Number of bins (mutually exclusive with -dlambda)")] = None,
    lmin: Annotated[Optional[float], typer.Option("-lmin", help="Minimum order parameter")] = None,
    lmax: Annotated[Optional[float], typer.Option("-lmax", help="Maximum order parameter")] = None,
    xcol: Annotated[int, typer.Option("-xcol", help="Order parameter column in order.txt")] = 1,
    dt: Annotated[Optional[float], typer.Option("-dt", help="Time step (auto-detected from toml if not set)")] = None,
    subcycles: Annotated[Optional[int], typer.Option("-subcycles", help="Number of MD subcycles per frame (auto-detected from toml if not set)")] = None,
    normalize: Annotated[str, typer.Option("-normalize", help="Normalization: none, time, density, or probability")] = "none",
    lm1: Annotated[bool, typer.Option("-lm1", help="Use lm1 correction for WHAM")] = True,
):
    """
    Compute histograms and free energies for an infretis simulation.
    
    Uses native inftools WHAM implementation (same as Wham_Pcross.py).
    
    Normalization options:
    - none: Raw histogram (weighted counts, sums to ~1 due to WHAM normalization)
    - time: Average time spent in each bin per trajectory (in time units of dt)
    - density: Probability density (integrates to 1 over order parameter range)
    - probability: Probability per bin (sum over bins equals 1; y-axis shows per-bin probability)

    Outputs CSV files:
    - wham_plus_histogram.csv: WHAM-weighted histogram for [i+] ensembles
    - ens_0min_histogram.csv: [0-] ensemble histogram
    - ens_NNN_histogram.csv: Per-ensemble histogram (without WHAM)
    """
    # Create output directory with robust handling
    outdir_path = Path(outdir)
    try:
        outdir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create output directory {outdir}: {e}")
        print("Using current directory instead...")
        outdir = "."
        outdir_path = Path(".")
    
    print(f"Computing histograms for simulation in {os.path.dirname(toml) or '.'}")
    print(f"Output directory: {outdir}")
    
    # Get interfaces from toml
    toml_dict = read_toml(toml)
    interfaces = [float(i) for i in toml_dict["simulation"]["interfaces"]]
    nintf = len(interfaces)
    nplus_ens = nintf - 1
    
    # Get dt and subcycles from toml if not provided
    if dt is None:
        dt = float(toml_dict.get("engine", {}).get("timestep", 1.0))
        print(f"Auto-detected dt={dt} from toml")
    if subcycles is None:
        subcycles = int(toml_dict.get("engine", {}).get("subcycles", 1))
        print(f"Auto-detected subcycles={subcycles} from toml")
    
    # Configuration
    i0plus, i0min = 4, 3
    lamres = 0.005
    lambdaA, lambdaB = interfaces[0], interfaces[-1]
    
    # Determine binning
    if dlambda is None and nbins is None:
        nbins = 100
    if dlambda is not None and nbins is not None:
        raise ValueError("Specify either -dlambda or -nbins, not both.")
    
    bin_lmin = lmin if lmin is not None else lambdaA - 0.3 * (lambdaB - lambdaA)
    bin_lmax = lmax if lmax is not None else lambdaB + 0.1 * (lambdaB - lambdaA)
    
    if nbins is not None:
        actual_nbins = nbins
        actual_dlambda = (bin_lmax - bin_lmin) / nbins
    else:
        if dlambda is None or dlambda <= 0:
            raise ValueError("Specify a positive -dlambda when -nbins is not provided")
        # Preserve exact user-supplied dlambda. Compute number of bins to
        # cover the requested range and extend the upper bound so that
        # bin edges are exact multiples of dlambda.
        actual_dlambda = dlambda
        actual_nbins = int(np.ceil((bin_lmax - bin_lmin) / actual_dlambda))
        if actual_nbins <= 0:
            raise ValueError("Computed number of bins is non-positive; check dlambda and range")
        # Adjust bin upper bound so the bin width remains exactly dlambda
        bin_lmax = bin_lmin + actual_nbins * actual_dlambda
    
    print(f"Binning: {actual_nbins} bins, width={actual_dlambda:.6f}")
    print(f"  Range: [{bin_lmin:.6f}, {bin_lmax:.6f}]")
    
    # Create histo_stuff dict (compatible with native inftools)
    histo_stuff = {
        "nbx": actual_nbins,
        "minx": bin_lmin,
        "maxx": bin_lmax,
        "xcol": xcol,
        "nby": None,
        "miny": None,
        "maxy": None,
        "ycol": None,
    }
    
    # ===== Read and process data matrix =====
    print("\n--- Reading data matrix ---")
    print(f"  Reading from: {data}")
    with open(data) as f:
        matrix = [
            [float(x) if x != "----" else 0.0 for x in line.strip().split()]
            for line in f if not line.startswith("#")
        ][nskip:]
    print(f"  Loaded {len(matrix)} paths (skipped first {nskip})")
    
    # Initialize eta and v_alpha
    print("\n--- Initializing WHAM computation ---")
    eta = [0.0] * nplus_ens
    lambda_values = [i * lamres for i in range(round(lambdaA / lamres), round(lambdaB / lamres) + 1)]
    v_alpha = [0.0] * len(lambda_values)
    v_alpha[0] = 1.0
    print(f"  Interfaces: {[f'{intf:.6f}' for intf in interfaces]}")
    print(f"  n_ensembles: {nintf}, n_plus_ensembles: {nplus_ens}")
    
    # Unweight matrix with HA-weights
    print("  Unweighting matrix with HA-weights...")
    sumPxy = [0.0] * nintf
    sumPxy_afterw = [0.0] * nintf
    
    for x in matrix:
        for y in range(nintf):
            y1, y2 = i0min + y, i0min + y + nintf
            P_xy = x[y1]
            sumPxy[y] += P_xy
            x[y1] = P_xy / x[y2] if x[y2] > 0 else 0.0
            sumPxy_afterw[y] += x[y1]
    
    # Normalize by average inverse HA-weight
    for y in range(nintf):
        if sumPxy[y] > 0:
            y1 = i0min + y
            AvinvwHA = sumPxy_afterw[y] / sumPxy[y]
            for x in matrix:
                x[y1] /= AvinvwHA
    
    # Compute eta and v_alpha from paths
    for x in matrix:
        lambdamax = x[2]
        for i in range(nplus_ens):
            Cxy = x[i0plus + i]
            eta[i] += Cxy
            
            # Determine bounds for increasing v_alpha values
            lambda_i = interfaces[i]
            alpha_max = int(np.floor((lambdamax - lambdaA) / lamres))
            alpha_min = round((lambda_i - lambdaA) / lamres)
            # Note: lambda_i-lambdaA)/lamres is an integer as
            # lambda_i and lambdaA should be commensurate with lamres
            if alpha_max > len(v_alpha) - 1:
                alpha_max = len(v_alpha) - 1  # -1 as we start counting from 0
            alpha_min += 1  # v(alpha) at the interface lambda_1, lambda_2
            # etc are determined by the previous [0+], [1+] etc
            for alpha in range(alpha_min, alpha_max + 1):
                v_alpha[alpha] += Cxy
    
    # Compute Q factors for WHAM
    def WHAM_PQ(npe, interf, res, eta, v_alpha):
        P, Q, invQ = [0.0] * npe, [0.0] * npe, [0.0] * npe
        P[0], invQ[0] = 1.0, eta[0]
        if invQ[0] == 0:
            return P, Q
        Q[0] = 1 / invQ[0]
        
        for i in range(1, npe):
            alpha = round((interf[i] - interf[0]) / res)
            # Ensure alpha doesn't exceed v_alpha bounds
            if alpha >= len(v_alpha):
                alpha = len(v_alpha) - 1
            P[i] = v_alpha[alpha] * Q[i - 1]
            if P[i] == 0:
                return P, Q
            invQ[i] = invQ[i - 1] + eta[i] / P[i]
            Q[i] = 1 / invQ[i]
        return P, Q
    
    # Compute Q factors using WHAM_PQ
    Pi0_wham, Q = WHAM_PQ(nplus_ens, interfaces, lamres, eta, v_alpha)
    
    # Get WHAM factors
    print("\n--- Computing WHAM factors ---")
    print(f"  eta (sampling per ensemble): {[f'{e:.2f}' for e in eta]}")
    print(f"  Q factors: {[f'{q:.6f}' for q in Q]}")
    print(f"  Pi0_wham (crossing probs): {[f'{p:.6e}' for p in Pi0_wham]}")
    
    WHAMfactors = get_WHAMfactors(matrix, interfaces, i0plus, Q, lm1)
    sum_wham = sum(WHAMfactors)
    WHAMfactors = [w / sum_wham for w in WHAMfactors] if sum_wham > 0 else WHAMfactors
    
    # Get [0-] ensemble weights
    WHAMfactors_0min = [x[i0min] for x in matrix]
    sum_0min = sum(WHAMfactors_0min)
    WHAMfactors_0min = [w / sum_0min for w in WHAMfactors_0min] if sum_0min > 0 else WHAMfactors_0min
    
    trajlabels = [int(x[0]) for x in matrix]
    dt_frame = dt * subcycles
    
    print(f"Processed {len(trajlabels)} paths")
    print(f"  [i+]: {sum(1 for w in WHAMfactors if w > 0)} paths with non-zero weight")
    print(f"  [0-]: {sum(1 for w in WHAMfactors_0min if w > 0)} paths with non-zero weight")
    print(f"  dt_frame = dt * subcycles = {dt} * {subcycles} = {dt_frame}")
    print(f"  Normalization mode: {normalize}")
    
    # ===== Compute histograms using native inftools functions =====
    
    # Cache for filtered path data to avoid re-reading trajectories
    path_cache = {}  # (path_nr, max_x) -> filtered trajectory array
    
    def get_filtered_trajectory(path_nr, max_x=None):
        """Load trajectory and optionally filter by max_x."""
        cache_key = (path_nr, max_x)
        if cache_key in path_cache:
            return path_cache[cache_key]
        
        trajfile = os.path.join(trajdir, str(path_nr), "order.txt")
        if not os.path.exists(trajfile):
            path_cache[cache_key] = np.array([])
            return path_cache[cache_key]
        
        try:
            traj_data = np.asarray(extract(trajfile, xcol))
            if max_x is not None:
                traj_data = traj_data[traj_data < max_x]
            path_cache[cache_key] = traj_data
            return traj_data
        except Exception as e:
            print(f"Warning: Could not process {trajfile}: {e}")
            path_cache[cache_key] = np.array([])
            return path_cache[cache_key]
    
    # Helper function to compute histogram with given weights
    def compute_histogram(weights, label, do_normalize=True, max_x_allowed=None):
        Nbinsx = histo_stuff["nbx"]
        Minx, Maxx = histo_stuff["minx"], histo_stuff["maxx"]
        dx = (Maxx - Minx) / Nbinsx
        edges = np.linspace(Minx, Maxx, Nbinsx + 1)
        xval = 0.5 * (edges[:-1] + edges[1:])
        histogram = np.zeros(Nbinsx)
        
        print(f"  {label}:")
        print(f"    Bin edges: [{Minx:.6f}, {Maxx:.6f}], Nbins={Nbinsx}, dx={dx:.6g}")
        if max_x_allowed is not None:
            print(f"    Filtering: keeping only frames with x < {max_x_allowed:.6f}")
        
        total_frames = 0.0  # weighted sum of filtered frames
        npaths_weight = 0.0  # sum of weights
        frames_filtered_out = 0.0  # weighted count of filtered frames
        
        for path_nr, weight in zip(trajlabels, weights):
            if weight == 0:
                continue
            
            traj_data = get_filtered_trajectory(path_nr, max_x_allowed)
            if len(traj_data) == 0:
                continue
            
            # Track filtering if applicable
            if max_x_allowed is not None:
                original_data = get_filtered_trajectory(path_nr, None)
                frames_filtered_out += weight * (len(original_data) - len(traj_data))
            
            # Filter to histogram range to prevent index out of bounds
            traj_data = traj_data[(traj_data >= Minx) & (traj_data < Maxx)]
            if len(traj_data) == 0:
                continue
            
            histogram = update_histogram(traj_data, weight, histogram, Minx, None, dx, None)
            total_frames += weight * len(traj_data)
            npaths_weight += weight
        
        # Mask forbidden bins
        if max_x_allowed is not None:
            mask = xval >= max_x_allowed
            n_masked_bins = np.sum(mask)
            weight_before_mask = np.sum(histogram[mask])
            histogram[mask] = 0.0
            if n_masked_bins > 0:
                print(f"    Masked {n_masked_bins} bins (x >= {max_x_allowed:.6f}), removed weight={weight_before_mask:.6g}")
            if frames_filtered_out > 0:
                print(f"    Filtered out {frames_filtered_out:.6g} weighted frames during pre-filtering")
        
        # Normalize histogram
        npaths = npaths_weight if npaths_weight > 0 else 1.0
        total_weight_sum = np.sum(histogram)
        
        print(f"    Effective paths: {npaths_weight:.6g}, weighted frames: {total_frames:.6g}")
        print(f"    Total histogram weight (after filtering/masking): {total_weight_sum:.6g}")
        
        if do_normalize and total_weight_sum > 0:
            if normalize == "time":
                histogram = histogram / total_weight_sum * total_frames * dt_frame / npaths / dx
                print(f"    Time normalization: hist = hist / {total_weight_sum:.3g} * {total_frames:.3g} * {dt_frame} / {npaths:.3g} / {dx:.6g}")
            elif normalize == "density":
                # Compute ensemble time (excluding first/last frames)
                time_ens = sum(
                    weight * max(0, len(get_filtered_trajectory(pn, max_x_allowed)) - 2)
                    for pn, weight in zip(trajlabels, weights) if weight > 0
                ) * dt_frame
                
                if time_ens > 0:
                    histogram = histogram / total_weight_sum * total_frames * dt_frame / npaths / dx / time_ens
                    print(f"    Density normalization: time_ens={time_ens:.6g}, result integrates to ~1")
                else:
                    histogram = histogram / total_weight_sum / dx
                    print(f"    Density normalization: time_ens=0, using hist / total_weight / dx")
                # Diagnostics: integral over x and maximum density
                integral = np.sum(histogram * dx)
                maxv = np.nanmax(histogram)
                print(f"    Density diagnostics: max={maxv:.6g}, integral={integral:.6g}, dx={dx:.6g}")
            elif normalize == "probability":
                # Probability per bin (sums to 1 across bins)
                histogram = histogram / total_weight_sum
                sumv = np.nansum(histogram)
                maxv = np.nanmax(histogram)
                print(f"    Probability normalization: sum={sumv:.6g}, max={maxv:.6g}")
        
        return xval, histogram
    
    # Helper function to compute free energy from histogram
    def compute_free_energy(histogram):
        max_value = np.max(histogram)
        if max_value > 0:
            prob = histogram / max_value
        else:
            prob = histogram.copy()
        
        with np.errstate(divide="ignore"):
            fe = -np.log(prob)
        fe[np.isinf(fe)] = np.nan
        
        return prob, fe
    
    # Determine column header
    hist_header = {
        "time": "avg_time_per_path",
        "density": "probability_density",
        "probability": "probability",
    }.get(normalize, "weighted_counts")
    
    # Helper to save histogram and free energy files
    def save_histogram_files(prefix, xval, histogram, prob, fe):
        hist_file = os.path.join(outdir, f"{prefix}_histogram.csv")
        fe_file = os.path.join(outdir, f"{prefix}_free_energy.csv")
        np.savetxt(hist_file, np.c_[xval, histogram, prob],
                   delimiter=",", header=f"order_parameter,{hist_header},probability", comments="")
        np.savetxt(fe_file, np.c_[xval, fe],
                   delimiter=",", header="order_parameter,free_energy_kBT", comments="")
        print(f"  Saved {prefix}_histogram.csv and {prefix}_free_energy.csv")
    
    # ===== WHAM [i+] histogram =====
    print("\n--- Computing WHAM [i+] histogram ---")
    xval, hist_wham_plus = compute_histogram(WHAMfactors, "WHAM [i+]")
    prob_wham_plus, fe_wham_plus = compute_free_energy(hist_wham_plus)
    save_histogram_files("wham_plus", xval, hist_wham_plus, prob_wham_plus, fe_wham_plus)
    
    # ===== [0-] ensemble histogram =====
    print("\n--- Computing [0-] ensemble histogram ---")
    xval, hist_0min = compute_histogram(WHAMfactors_0min, "[0-]", max_x_allowed=interfaces[0])
    prob_0min, fe_0min = compute_free_energy(hist_0min)
    save_histogram_files("ens_0min", xval, hist_0min, prob_0min, fe_0min)

    # ===== Per-ensemble histograms (without WHAM) =====
    print("\n--- Computing per-ensemble histograms (without WHAM) ---")
    
    # Build ensemble paths and weights from data file
    paths = data_reader(data)[nskip:]
    ens_paths = {i: [] for i in range(nintf)}
    ens_weights = {i: [] for i in range(nintf)}
    print(f"  Building ensemble path lists from {len(paths)} paths...")
    
    for path in paths:
        pn = int(path["pn"])
        for col, (f0, w0) in path["cols"].items():
            f0, w0 = float(f0), float(w0)
            if w0 > 0:
                ens_paths[col].append(pn)
                ens_weights[col].append(f0 / w0)
    
    # Normalize weights per ensemble
    for i in range(nintf):
        w = np.array(ens_weights[i])
        if len(w) > 0 and np.sum(w) > 0:
            ens_weights[i] = w / np.sum(w)
            ens_label = f"[{i}-]" if i == 0 else f"[{i-1}+]"
            print(f"  Ensemble {i} ({ens_label}): {len(w)} paths, sum(weights)={np.sum(ens_weights[i]):.6f}")
    
    all_ens_hists = []
    all_ens_fes = []
    headers_hist = []
    headers_fe = []
    
    # Helper to compute histogram for a specific ensemble
    def compute_ensemble_histogram(ens_idx, path_nrs, weights):
        ens_label = f"[{ens_idx}-]" if ens_idx == 0 else f"[{ens_idx-1}+]"
        max_x = interfaces[0] if ens_idx == 0 else None
        
        # Reuse main histogram computation logic
        xval, histogram = compute_histogram(weights, f"Ensemble {ens_idx} ({ens_label})", do_normalize=True, max_x_allowed=max_x)
        prob, fe = compute_free_energy(histogram)
        
        # Save files
        np.savetxt(
            os.path.join(outdir, f"ens_{ens_idx:03d}_histogram.csv"),
            np.c_[xval, histogram, prob],
            delimiter=",",
            header=f"order_parameter,{hist_header}_{ens_label},probability_{ens_label}",
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, f"ens_{ens_idx:03d}_free_energy.csv"),
            np.c_[xval, fe],
            delimiter=",",
            header=f"order_parameter,free_energy_{ens_label}_kBT",
            comments="",
        )
        
        return histogram, fe, ens_label
    
    for ens_idx in range(nintf):
        if len(ens_paths[ens_idx]) == 0:
            ens_label = f"[{ens_idx}-]" if ens_idx == 0 else f"[{ens_idx-1}+]"
            print(f"  Ensemble {ens_idx} ({ens_label}): no paths")
            continue
        
        # Temporarily swap trajlabels to use ensemble-specific paths
        orig_trajlabels = trajlabels
        trajlabels = ens_paths[ens_idx]
        
        hist, fe, label = compute_ensemble_histogram(ens_idx, ens_paths[ens_idx], ens_weights[ens_idx])
        
        trajlabels = orig_trajlabels  # Restore
        
        all_ens_hists.append(hist)
        all_ens_fes.append(fe)
        headers_hist.append(label)
        headers_fe.append(label)
    
    # Save combined per-ensemble data
    if all_ens_hists:
        np.savetxt(
            os.path.join(outdir, "all_ensembles_histogram.csv"),
            np.column_stack([xval] + all_ens_hists),
            delimiter=",", header="order_parameter," + ",".join(headers_hist), comments="",
        )
        np.savetxt(
            os.path.join(outdir, "all_ensembles_free_energy.csv"),
            np.column_stack([xval] + all_ens_fes),
            delimiter=",", header="order_parameter," + ",".join(headers_fe), comments="",
        )
        print(f"  Saved combined data with {len(all_ens_hists)} ensembles")
    
    print("\n--- Done ---")
    print(f"All output files saved to: {outdir}")
    return {"outdir": outdir, "interfaces": interfaces, "n_paths": len(trajlabels)}


if __name__ == "__main__":
    typer.run(compute_all_histograms)
