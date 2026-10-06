#!/usr/bin/env python3
"""
error_analysis_staple: running estimate + recursive block error analysis for
iSTAR/StapleTIS simulations run with infretis.

Reads `infretis_data.txt` + `infretis.toml` (same front end as
`tistools/scripts/infretis_memory_analysis.py`), computes running estimates
of P_cross, the Q and P (MSM) matrices (and optionally flux/MFPT/rate) every
`interval` paths, and performs vectorized recursive block error analysis on them.

Outputs (in --outdir, default <simdir>):
  - pcross_runav.txt, qmat_runav.txt, pmat_runav.txt, rate_runav.txt
  - {pcross,qmat,pmat,rate}_block_errors_<interval>.{txt,png}

Usage:
    inft error_analysis_staple <simulation_dir> [options]

Example:
    inft error_analysis_staple /path/to/infretis_sim --interval 100 --nskip 1000
"""

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path
import numpy as np
from typing import Annotated, Dict, Optional

import typer


# =============================================================================
# PROGRESS / OUTPUT HELPERS
# =============================================================================
@contextlib.contextmanager
def quiet(enabled=True):
    """Silence the very chatty weight/tistools analysis routines."""
    if not enabled:
        yield None
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield buf


def _fmt_time(seconds):
    if not np.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def make_progress(cycles, stream=sys.stderr, min_dt=0.2, work=None, unit="steps", label="cycle"):
    """
    Progress reporter for the running-estimate loop (and the path-length pass).

    The cost of each step grows with the number of paths it covers (every
    snapshot re-scans the paths from the start), so the ETA is weighted by
    paths done / paths total rather than by step count -- a step-count ETA
    would be wildly optimistic early on. Redraws are throttled to one per
    `min_dt` seconds, since with small intervals there can be many thousands
    of steps. Pass `work` to weight the ETA by something other than `cycles`
    (e.g. uniform per-path cost), and `unit`/`label` to relabel the bar.
    """
    cycles = np.asarray(cycles)
    work = np.cumsum(np.asarray(cycles if work is None else work, dtype=float))
    total = len(cycles)
    state = {"t0": time.time(), "last": -np.inf}

    def progress(idx):
        now = time.time()
        final = idx + 1 == total
        if not final and now - state["last"] < min_dt:
            return
        state["last"] = now
        frac = min(work[idx] / work[-1], 1.0) if work[-1] > 0 else 1.0
        elapsed = now - state["t0"]
        eta = elapsed / frac - elapsed if frac > 1e-9 else float("inf")
        width = 30
        filled = int(width * frac)
        bar = "#" * filled + "." * (width - filled)
        stream.write(f"\r  [{bar}] {idx + 1:>4}/{total} {unit}  {label} {int(cycles[idx]):>9}  "
                     f"elapsed {_fmt_time(elapsed):>7}  eta {_fmt_time(eta):>7}  ")
        stream.flush()
        if final:
            stream.write("\n")
            stream.flush()

    return progress


# =============================================================================
# OPTIMIZED BLOCK ERROR FUNCTIONS (From Notebook)
# =============================================================================
def calculate_infretis_weights(data_file: str, toml_file: str, nskip: int = 0) -> Dict:
    """
    Calculate infretis weights for paths, based on the path_weights.py methodology.
    Updated to use data_reader from inftools.misc.data_helper.
    
    Parameters:
    -----------
    data_file : str
        Path to infretis_data.txt file
    toml_file : str  
        Path to infretis.toml configuration file
    nskip : int
        Number of initial entries to skip
        
    Returns:
    --------
    Dict containing path data and weights
    """
    try:
        import tomllib as tomli
    except ImportError:
        import tomli
    import re
    import random
    from inftools.misc.data_helper import data_reader
    
    def parse_ptype_direction(ptype):
        """
        Parse ptype to extract direction based on interface indices.
        Uses the exact logic from conv_inf_py.py
        """
        # Handle simple ptype formats (ensemble 0)
        if ptype in ['RMR', 'RML', 'LMR', 'LML', 'L*L', 'R*R']:
            return 1
        
        # Handle complex ptype formats with interface indices
        # Pattern to match: digits + letters + digits
        match = re.match(r'^(\d+)([LR]M[LR])(\d+)$', ptype)
        if match:
            try:
                a = int(match.group(1))  # First interface index
                b = int(match.group(3))  # Last interface index
                
                if a <= b:
                    return 1
                elif a > b:
                    return -1
            except ValueError:
                # If parsing fails, default to 1
                return 1
        
        # Default case
        return 1

    def extract_ptype_middle(ptype):
        """
        Extract the middle part (XMX) from ptype.
        Uses the exact logic from conv_inf_py.py
        """
        # Handle simple ptype formats (ensemble 0)
        if ptype in ['RMR', 'RML', 'LMR', 'LML', 'L*L', 'R*R']:
            return ptype
        
        # Handle complex ptype formats with interface indices
        match = re.match(r'^(\d+)([LR]M[LR])(\d+)$', ptype)
        if match:
            return match.group(2)  # Return the middle part (XMX)
        
        # Default case - return as is
        return ptype
    
    # Load configuration
    with open(toml_file, "rb") as f:
        toml_config = tomli.load(f)
    interfaces = toml_config["simulation"]["interfaces"]
    # read lm1 (lambda_minus_one) if present in toml
    lm1 = toml_config.get("simulation", {}).get("tis_set", {}).get("lambda_minus_one", None)
    # infretis writes `lambda_minus_one = false` when it is disabled
    if isinstance(lm1, bool):
        lm1 = None
    if lm1 is not None:
        print(f"read lm1 from toml: {lm1}")

    
    # infretis_data.txt has 2*n_interfaces + 5 columns (pnr, len, maxop, minop,
    # ptype, then path_f and path_w per interface). Some rows in practice have
    # extra trailing whitespace-separated junk, which makes a plain
    # np.loadtxt(..., dtype=str) (no usecols) see a ragged/inconsistent column
    # count and fail; pin usecols to the expected width to avoid that.
    n_cols = 2 * len(interfaces) + 5
    data = np.loadtxt(data_file, dtype=str, usecols=np.arange(n_cols))
    data = data[nskip:]  # Skip initial entries
    
    # Check if we have ptype information (look for correct patterns)
    has_ptype = False
    ptype_col = None
    
    # Look for ptype patterns in the data - use correct patterns
    for col in range(data.shape[1]):
        sample_values = data[:10, col]  # Check first 10 rows
        for val in sample_values:
            if isinstance(val, str) and (re.match(r'\d+[LR]M[LR]\d+', val) or val in ['RMR', 'RML', 'LMR', 'LML', 'L*L', 'R*R']):
                has_ptype = True
                ptype_col = col
                break
        if has_ptype:
            break
    
    if has_ptype:
        print(f"Found ptype information in column {ptype_col}")
        # Extract direction information from ptype
        directions = []
        start_interfaces = []
        end_interfaces = []
        
        for i, row in enumerate(data):
            ptype = row[ptype_col]
            if isinstance(ptype, str):
                # Parse using the correct logic from conv_inf_py.py
                direction = parse_ptype_direction(ptype)
                directions.append(direction)
                
                # Extract start and end interface indices
                if ptype in ['RMR', 'RML', 'LMR', 'LML', 'L*L', 'R*R']:
                    # Simple format - ensemble 0
                    start_interfaces.append(-1)
                    end_interfaces.append(0)
                else:
                    # Complex format with interface indices
                    match = re.match(r'^(\d+)([LR]M[LR])(\d+)$', ptype)
                    if match:
                        a = int(match.group(1))  # start interface index
                        b = int(match.group(3))  # end interface index
                        start_interfaces.append(a)
                        end_interfaces.append(b)
                    else:
                        start_interfaces.append(0)
                        end_interfaces.append(0)
            else:
                directions.append(0)  # Default for non-ptype entries
                start_interfaces.append(0)
                end_interfaces.append(0)
    else:
        print("No ptype information found, using standard format")
        directions = []
        start_interfaces = []
        end_interfaces = []
    
    # Identify non-zero paths (those with "----" in the zero-ensemble column)
    # Adjust column index based on whether ptype is present
    zero_col = 4 if not has_ptype else 5  # Assumes ptype is typically after maxop
    if zero_col < data.shape[1]:
        non_zero_paths = data[:, zero_col] == "----"
    else:
        # Fallback: look for "----" in any column after maxop
        non_zero_paths = data[:, 3] == "----"
    
    # Replace "----" with "0.0" for numerical processing
    data[data == "----"] = "0.0"
    non_zero_paths = np.full_like(non_zero_paths, True)
    
    # Extract path information
    D = {}
    D["pnr"] = data[non_zero_paths, 0:1].astype(int)  # Path numbers
    D["len"] = data[non_zero_paths, 1:2].astype(int)  # Path lengths
    D["maxop"] = data[non_zero_paths, 2:3].astype(float)  # Maximum order parameter
    D['minop'] = data[non_zero_paths, 3:4].astype(float)
    
    # Add ptype-derived information if available
    if has_ptype:
        D["ptype"] = data[non_zero_paths, ptype_col]  # Path types
        D["direction"] = np.array([directions[i] for i in range(len(directions)) if non_zero_paths[i]])
        D["start_intf"] = np.array([start_interfaces[i] for i in range(len(start_interfaces)) if non_zero_paths[i]])
        D["end_intf"] = np.array([end_interfaces[i] for i in range(len(end_interfaces)) if non_zero_paths[i]])
    
    # Determine data columns for path_f and path_w
    data_start_col = ptype_col + 1 if has_ptype else 4
    D["path_f"] = data[non_zero_paths, data_start_col : data_start_col + len(interfaces)].astype(float)  # Path occurrences
    D["path_w"] = data[non_zero_paths, data_start_col + len(interfaces) : data_start_col + 2 * len(interfaces)].astype(float)  # Path weights
    
    # Calculate weights w = path_f / path_w
    w = D["path_f"] / D["path_w"]
    w[np.isnan(w)] = 0
    
    # Normalize weights to match total number of samples
    w = w / np.sum(w, axis=0) * np.sum(D["path_f"], axis=0)
    w[np.isnan(w)] = 0.0
    wsum = np.sum(w, axis=0)
    
    # # Calculate local crossing probabilities (ploc) using WHAM
    # ploc_wham = np.zeros(len(interfaces))
    # ploc_wham[0] = 1.0
    
    # for i, intf_p1 in enumerate(interfaces[1:]):
    #     h1 = D["maxop"] >= intf_p1
    #     nj = wsum[:i + 1]  # Number of paths crossing lambda_i for each ensemble up to i
    #     njl = np.sum(h1 * w[:, : i + 1], axis=0)  # Number of paths crossing lambda_i+1
    #     ploc_wham[i + 1] = np.sum(njl) / np.sum(nj / ploc_wham[: i + 1])
    
    # # Calculate unbiased path weights
    # A = np.zeros_like(D["maxop"])
    # Q = 1 / np.cumsum(wsum / ploc_wham[:-1])
    
    # for j, pathnr in enumerate(D["pnr"][:, 0]):
    #     # Find the highest interface crossed by this path
    #     K = min(
    #         np.where(D["maxop"][j] > interfaces)[0][-1] if np.any(D["maxop"][j] > interfaces) else 0, 
    #         len(interfaces) - 2
    #     )
    #     A[j] = Q[K] * np.sum(w[j])
    
    # Store results
    results = {
        'interfaces': interfaces,
        'path_data': D,
        'weights_matrix': w,
        # 'unbiased_weights': A,
        # 'ploc_wham': ploc_wham,
        # 'Q_factors': Q,
        'has_ptype': has_ptype,
        'lm1': lm1,
    }
    
    print(f"Processed {len(D['pnr'])} paths")
    print(f"Interfaces: {interfaces}")
    # print(f"Local crossing probabilities (WHAM): {ploc_wham}")
    
    if has_ptype:
        print(f"Path type information detected:")
        print(f"  Forward paths (dir=1): {np.sum(D['direction'] == 1)}")
        print(f"  Backward paths (dir=-1): {np.sum(D['direction'] == -1)}")
        print(f"  Other paths (dir=0): {np.sum(D['direction'] == 0)}")
    
    return results

def compute_weight_matrices_weights(weight_results: Dict, n_int: Optional[int] = None, tr: bool = True) -> Dict:
    """
    Compute 3D weight matrices from path data following original istar_analysis logic.
    
    This function constructs weight matrices [i,j,k] where:
    - i: ensemble index (path ensemble)
    - j: starting interface index
    - k: ending interface index
    
    Implements the original istar_analysis.py logic for:
    - tr (time reversal): boolean for applying time-reversal symmetry
    - Edge case handling for specific interface transitions
    - Proper direction mask logic and boundary conditions
    
    Parameters:
    -----------
    weight_results : Dict
        Results from calculate_infretis_weights function containing:
        - path_data (D): Dictionary with path information including path_f and path_w
        - interfaces: List of interface positions
        - has_ptype: Boolean indicating if ptype information is available
    tr : bool, optional
        If True, applies time-reversal symmetry by symmetrizing weight matrices.
        Default is False.
        
    Returns:
    --------
    Dict containing:
        - weight_matrix_3d: 3D array [i,j,k] with weights for ensemble i, from interface j to k
        - count_matrix_3d: 3D array [i,j,k] with path counts for ensemble i, from interface j to k
        - weight_matrix_2d: 2D array [j,k] with total weights (summed over ensembles)
        - count_matrix_2d: 2D array [j,k] with total count (summed over ensembles)
        - ensemble_totals: 1D array with total weights per ensemble
        - transition_summary: Dictionary with detailed transition statistics
        - tr_applied: Boolean indicating if time-reversal symmetry was applied
    """
    
    # Extract data from weight_results
    D = weight_results['path_data']
    if n_int is None:
        interfaces = weight_results['interfaces']
    else:
        interfaces = weight_results['interfaces'][:n_int]
    has_ptype = weight_results.get('has_ptype', False)
    
    n_interfaces = len(interfaces)
    n_ensembles = len(interfaces)  # Number of ensembles equals number of interfaces
    n_paths = len(D['pnr'])
    
    print(f"Computing weight matrices for {n_paths} paths")
    print(f"Structure: Dictionary of 2D matrices [ensemble_idx][start_interface, end_interface]")
    print(f"Dimensions: {n_ensembles} ensembles x {n_interfaces} interfaces x {n_interfaces} interfaces")
    print(f"Following original istar_analysis.py logic with tr={tr}")
    
    # Initialize dictionaries of 2D matrices {ensemble_i: [start_interface_j, end_interface_k]}
    weight_matrix_3d = {i: np.zeros((n_interfaces, n_interfaces)) for i in range(n_ensembles)}
    weight_matrix_3d_norm = {i: np.zeros((n_interfaces, n_interfaces)) for i in range(n_ensembles)}
    count_matrix_3d = {i: np.zeros((n_interfaces, n_interfaces)) for i in range(n_ensembles)}
    
    # Arrays to track totals
    ensemble_totals = np.zeros(n_ensembles)
    
    if has_ptype and 'start_intf' in D and 'end_intf' in D and 'direction' in D:
        print("Using ptype information with direction for istar_analysis-style computation")
        
        # Pre-compute the normalization factor once (moved outside the loop for efficiency)
        # This computes: sum(path_f) / sum(path_f / path_w) for each ensemble
        normalization_factor = np.nan_to_num(
            np.sum(D['path_f'], axis=0) / np.sum(np.nan_to_num(D['path_f'] / D['path_w']), axis=0)
        )
        
        # Process each path
        for path_idx in range(n_paths):
            ptype = D['ptype'][path_idx]
            start_intf = int(D['start_intf'][path_idx])
            end_intf = int(D['end_intf'][path_idx])
            direction = int(D['direction'][path_idx])  # 1 for forward, -1 for backward, 0 for other
            
            # Validate interface indices
            if (((start_intf < 0 or start_intf >= n_interfaces) and
                (end_intf < 0 or end_intf >= n_interfaces)) or
                (start_intf >= n_interfaces-1 and end_intf >= n_interfaces-1 and n_int is not None)):
                continue

            start_intf = min(start_intf, n_interfaces - 1)  # Ensure within bounds
            end_intf = min(end_intf, n_interfaces - 1)  # Ensure within bounds
                
            # Process each ensemble for this path (following original istar_analysis logic)
            # Calculate weight as path_f / path_w
            path_f_k = D['path_f'][path_idx, :]
            path_w_k = np.array([min(D['path_w'][path_idx, i], 1.) for i in range(len(D['path_w'][path_idx, :]))])  # TODO: why min()?
            weight_k = np.nan_to_num(path_f_k / path_w_k) if np.sum(path_w_k) != 0 else np.zeros_like(path_f_k)
            weight_k *= normalization_factor  # Use pre-computed normalization factor
            # DONT ENABLE NORMALIZATION PER ROW, wrong results
            # weight_k = np.nan_to_num(weight_k / np.sum(weight_k) * np.sum(path_f_k)) if (np.sum(path_w_k) != 0 or np.sum(path_f_k) != 0) else 0  # TODO: normalization?

            if weight_k[0] == 0:
                for i in range(1,n_ensembles):
                    if np.sum(path_w_k) != 0 and np.sum(path_f_k) != 0:  # Only process non-zero entries
                    # if weight_results['weights_matrix'][path_idx, i] != 0:
                        # weight = weight_results['weights_matrix'][path_idx, i]
                        weight = weight_k[i]
                        # assert(weight == weight_results['weights_matrix'][path_idx, i]), f"Weight mismatch for path {path_idx}, ensemble {i}: {weight} != {weight_results['weights_matrix'][path_idx, i]}"
                        # if not tr and ((i == 2 and "LML" in ptype) or (i == len(interfaces) - 1 and "RMR" in ptype)):
                        #     weight /= 2
                        #     if (i == 2 and "LML" in ptype):
                        #         weight /= 2  # Additional halving for LML in ensemble 2
                        # Apply original istar_analysis logic for j→k transitions
                        j, k = start_intf, end_intf
                        
                        # Determine if this path should be counted in ensemble i
                        should_count = False
                        
                        if weight == 0:
                            continue
                        
                        if j == k:
                            # Self-transitions: Special case for i==1 (ensemble 1) and j==0
                            if j == 0:
                                # Original logic: count LMR paths in ensemble 1 for 0→0 transitions
                                # if 'LMR' in D['ptype'][path_idx] or ('LML' in D['ptype'][path_idx] and D['maxop'][path_idx] >= interfaces[1]):
                                #     k = 1  # Adjust to next interface for ensemble 1
                                should_count = True
                            elif j == len(interfaces) - 1:
                                # should not happen with new implementation
                                # print("nooooo")
                                k = len(interfaces) - 2  # Last interface self-transition
                                should_count = True  # Original logic: count RMR paths in last ensemble
                            else:
                                print(j,k, ptype)
                                should_count = False
                                
                        elif j < k:
                            # Forward transitions (j → k where j < k)
                            
                            # Edge case 1: j==0 and k==1 (first interface to second)
                            if j == 0 and k == 1:
                                # print("shouldnt happen first")
                                if i != 2:
                                    # Use direction==1 for forward paths
                                    should_count = (direction == 1)
                                    assert should_count
                                elif i == 2:
                                    # Special case: ensemble 2 uses different logic
                                    # In original: dir_mask = masks[i]["LML"]
                                    should_count = True  # Simplified - would need LML mask
                                    
                            # Edge case 2: Last interface transition
                            elif j == len(interfaces)-2 and k == len(interfaces)-1:
                                # print("shouldnt happen last")
                                # Original: dir_mask = masks[i]["RMR"]
                                should_count = True  # Simplified - would need RMR mask

                            elif i-1 in [j, k] and 1 < i < len(interfaces):
                                # print(f"path_w: {D['path_w'][path_idx, i]}, path_f: {D['path_f'][path_idx, i]}, weight: {weight}, j: {j}, k: {k}, i: {i}, ptype: {ptype}")
                                # weight *= 2
                                should_count = True
                            else:
                                # Standard forward transitions
                                should_count = (direction == 1)
                                assert should_count
                                
                        else:
                            # Backward transitions (j → k where j > k)
                            
                            # Edge case 1: j==1 and k==0 (second interface to first)
                            if j == 1 and k == 0:
                                # print("shouldnt happen first backward")
                                if i != 2:
                                    # Use direction==-1 for backward paths
                                    should_count = (direction == -1)
                                    assert should_count
                                elif i == 2:
                                    # Special case: ensemble 2 uses different logic
                                    should_count = True  # Simplified - would need LML mask
                                    
                            # Edge case 2: Last interface backward transition
                            elif j == len(interfaces)-1 and k == len(interfaces)-2:
                                # print("shouldnt happen last backward")
                                # Original: dir_mask = masks[i]["RMR"]
                                should_count = True  # Simplified - would need RMR mask
                                
                            elif i-1 in [j, k] and 1 < i < len(interfaces):
                                # print(f"path_w: {D['path_w'][path_idx, i]}, path_f: {D['path_f'][path_idx, i]}, weight: {weight}, j: {j}, k: {k}, i: {i}, ptype: {ptype}")
                                # weight *= 2
                                should_count = True
                            else:
                                # Standard backward transitions
                                should_count = (direction == -1)
                                assert should_count
                        
                        # Count the transition if criteria are met
                        if should_count:
                            
                            weight_matrix_3d[i][j, k] += weight
                            count_matrix_3d[i][j, k] += 1
                            ensemble_totals[i] += weight
            else:
                weight = weight_k[0]
                weight_matrix_3d[0][0, 0] += weight
                count_matrix_3d[0][0, 0] += 1
                ensemble_totals[0] += weight
    
    else:
        print("No ptype information with direction available")
        raise ValueError("Path data must contain 'start_intf', 'end_intf', and 'direction' for istar_analysis-style computation.")
    
    # Apply time-reversal symmetry if requested (following original istar_analysis logic)
    weight_matrix_3d_notr = {i: weight_matrix_3d[i].copy() for i in range(n_ensembles)}
    count_matrix_3d_notr = {i: count_matrix_3d[i].copy() for i in range(n_ensembles)}
    weight_matrix_2d_notr = np.zeros((n_interfaces, n_interfaces))
    count_matrix_2d_notr = np.zeros((n_interfaces, n_interfaces))
    if tr:
        print("Applying time-reversal symmetry (tr=True)")
        
        for i in range(n_ensembles):
            # Original edge case logic for time reversal
            # if i == 2 and weight_matrix_3d[i][1, 0] == 0:
            #     # In [1*] all LML paths are classified as 1 → 0 (for now).
            #     # Time reversal needs to be adjusted to compensate for this
            #     weight_matrix_3d[i][0, 1] *= 2
            #     print(f"  Applied tr edge case for ensemble 2: doubled weight_matrix_3d[{i}, 0, 1]")
                
            # elif i == len(interfaces)-1 and weight_matrix_3d[i][-2, -1] == 0:
            #     weight_matrix_3d[i][-1, -2] *= 2
            #     print(f"  Applied tr edge case for last ensemble: doubled weight_matrix_3d[{i}, -1, -2]")
            
            # Properly symmetrize the matrix: X[i] = (X[i] + X[i].T) / 2.0
            weight_matrix_3d[i] = (weight_matrix_3d[i] + weight_matrix_3d[i].T) / 2.0
            count_matrix_3d[i] = (count_matrix_3d[i] + count_matrix_3d[i].T) / 2.0
    
    # Calculate 2D matrices by summing over ensembles
    weight_matrix_2d = np.zeros((n_interfaces, n_interfaces))
    count_matrix_2d = np.zeros((n_interfaces, n_interfaces))

    for i in range(n_ensembles):
        weight_matrix_2d += weight_matrix_3d[i]
        count_matrix_2d += count_matrix_3d[i]
        weight_matrix_2d_notr += weight_matrix_3d_notr[i]
        count_matrix_2d_notr += count_matrix_3d_notr[i]
    
    # Create transition summary
    total_weight = sum(np.sum(weight_matrix_3d[i]) for i in range(n_ensembles))
    total_transitions = sum(np.sum(count_matrix_3d[i]) for i in range(n_ensembles))
    
    # Analyze transition types across all ensembles
    forward_transitions = 0
    backward_transitions = 0
    self_transitions = 0
    forward_weight = 0
    backward_weight = 0
    self_weight = 0 
    
    for i in range(n_ensembles):
        for j in range(n_interfaces):
            for k in range(n_interfaces):
                weight_ijk = weight_matrix_3d[i][j, k]
                count_ijk = count_matrix_3d[i][j, k]
                
                if count_ijk > 0:
                    if j < k:  # Forward transition
                        forward_transitions += count_ijk
                        forward_weight += weight_ijk
                    elif j > k:  # Backward transition
                        backward_transitions += count_ijk
                        backward_weight += weight_ijk
                    else:  # Self transition
                        self_transitions += count_ijk
                        self_weight += weight_ijk
    
    transition_summary = {
        'total_weight': total_weight,
        'total_transitions': total_transitions,
        'forward_transitions': forward_transitions,
        'backward_transitions': backward_transitions,
        'self_transitions': self_transitions,
        'forward_weight': forward_weight,
        'backward_weight': backward_weight,
        'self_weight': self_weight,
        'forward_weight_fraction': forward_weight / total_weight if total_weight > 0 else 0,
        'backward_weight_fraction': backward_weight / total_weight if total_weight > 0 else 0,
        'self_weight_fraction': self_weight / total_weight if total_weight > 0 else 0
    }
    
    # Print detailed results following original istar_analysis style
    print(f"\n=== 3D WEIGHT MATRICES RESULTS (istar_analysis style) ===")
    print(f"3D Matrix dimensions: {n_ensembles} x {n_interfaces} x {n_interfaces}")
    print(f"Total weight processed: {total_weight:.6f}")
    print(f"Total transitions: {total_transitions}")
    print(f"Non-zero 3D matrix elements: {sum(np.count_nonzero(weight_matrix_3d[i]) for i in range(n_ensembles))}")
    print(f"Time-reversal symmetry applied: {tr}")
    
    # Print ensemble weights (like original "Sum weights ensemble i")
    print(f"\nEnsemble weight totals:")
    for i in range(n_ensembles):
        ensemble_sum = np.sum(weight_matrix_3d[i])
        print(f"  Sum weights ensemble {i}: {ensemble_sum:.4f}")
    
    print(f"\n2D Weight Matrix [start_interface, end_interface] (summed over ensembles):")
    print("Rows = start interface, Columns = end interface")
    for j in range(n_interfaces):
        row_str = f"Interface {j}: "
        for k in range(n_interfaces):
            row_str += f"{weight_matrix_2d[j, k]:8.4f} "
        print(row_str)
    
    print(f"\nTransition Analysis:")
    print(f"Forward transitions (j<k):  {forward_transitions:4f} paths, {forward_weight:8.4f} weight ({transition_summary['forward_weight_fraction']:.1%})")
    print(f"Backward transitions (j>k): {backward_transitions:4f} paths, {backward_weight:8.4f} weight ({transition_summary['backward_weight_fraction']:.1%})")
    print(f"Self transitions (j=k):     {self_transitions:4f} paths, {self_weight:8.4f} weight ({transition_summary['self_weight_fraction']:.1%})")

    # Show some 3D matrix details for non-zero entries
    print(f"\nNon-zero 3D matrix entries (first 10):")
    count = 0
    for i in range(n_ensembles):
        for j in range(n_interfaces):
            for k in range(n_interfaces):
                if weight_matrix_3d[i][j, k] > 0 and count < 10:
                    print(f"  weights[{i},{j},{k}] = {weight_matrix_3d[i][j, k]:.6f} (count: {count_matrix_3d[i][j, k]:.1f})")
                    count += 1
                if count >= 10:
                    break
            if count >= 10:
                break
        if count >= 10:
            break
    
    # Store and return results
    results = {
        'weight_matrix_3d': weight_matrix_3d,
        'count_matrix_3d': count_matrix_3d,
        'weight_matrix_3d_notr': weight_matrix_3d_notr,
        'count_matrix_3d_notr': count_matrix_3d_notr,
        'weight_matrix_2d': weight_matrix_2d,
        'count_matrix_2d': count_matrix_2d,
        'weight_matrix_2d_notr': weight_matrix_2d_notr,
        'count_matrix_2d_notr': count_matrix_2d_notr,
        'ensemble_totals': ensemble_totals,
        'transition_summary': transition_summary,
        'tr_applied': tr,
        'interfaces': interfaces,
        'n_interfaces': n_interfaces,
        'n_ensembles': n_ensembles,
        'total_paths_processed': n_paths
    }
    
    return results

def block_error(data, maxblock=None, blockskip=1):
    """Block error analysis via numpy reshape — no Python per-element loop."""
    n = len(data)
    maxblock = min(maxblock or n // 2, n // 2)
    blocklen = np.arange(1, maxblock + 1, blockskip, dtype=np.intp)

    block_avg = np.empty(len(blocklen))
    block_err = np.empty(len(blocklen))

    for i, b in enumerate(blocklen):
        n_full = (n // b) * b
        bm = data[:n_full].reshape(-1, b).mean(axis=1)
        nb_i = len(bm)
        block_avg[i] = bm.mean()
        block_err[i] = bm.std(ddof=1) / np.sqrt(nb_i) if nb_i > 1 else 0.0

    large_blocks = blocklen > maxblock // 2
    block_err_avg = (np.mean(block_err[large_blocks])
                     if np.any(large_blocks) else block_err[-1])
    return blocklen, block_avg, block_err, block_err_avg, maxblock, n // maxblock

def rec_blocks_from_runav(runav, n):
    """Reconstruct block averages from a running-average time series."""
    assert n > 0
    runav_red = runav[n - 1::n]
    nb = len(runav_red)
    idx  = np.arange(nb, dtype=float)
    prev = np.empty(nb)
    prev[0]  = 0.0
    prev[1:] = runav_red[:-1]
    return (idx + 1) * runav_red - idx * prev

def compute_rel_errors_2d(runavfull, sizes, bestav=None):
    """Vectorized block error for 2D running-average data (processes all elements at once)."""
    flat = runavfull.ndim == 1
    if flat:
        runavfull = runavfull[:, None]
    if bestav is None:
        bestav = runavfull[-1]
    bestav      = np.asarray(bestav).ravel()
    safe_bestav = np.where(bestav != 0, np.abs(bestav), 1.0)

    n_features = runavfull.shape[1]
    rel_errors = np.empty((len(sizes), n_features))

    for i, n in enumerate(sizes):
        rr  = runavfull[n - 1::n, :]
        nb_i = rr.shape[0]
        if nb_i < 2:
            rel_errors[i, :] = 0.0
            continue
        idx  = np.arange(nb_i, dtype=float)[:, None]
        prev = np.empty_like(rr)
        prev[0]  = 0.0
        prev[1:] = rr[:-1]
        blocks   = (idx + 1) * rr - idx * prev
        sq_diff  = np.sum((blocks - bestav[None, :]) ** 2, axis=0)
        Aerr     = np.sqrt(sq_diff / (nb_i * (nb_i - 1)))
        rel_errors[i, :] = Aerr / safe_bestav

    return rel_errors[:, 0] if flat else rel_errors



# =============================================================================
# PATH LENGTHS (ported from infretis_istar_workflow.ipynb)
# =============================================================================
def load_order_from_path(path_dir):
    """Load the order parameters of one path from <path_dir>/order.txt (time column dropped)."""
    order_file = Path(path_dir) / "order.txt"
    if not order_file.exists():
        return None
    try:
        data = np.loadtxt(order_file, comments='#')
    except Exception:
        return None
    if data.ndim == 1:
        # A single line: [time, op, ...]
        data = data.reshape(1, -1)
    return data[:, 1:] if data.shape[1] > 1 else data


def compute_path_taus(weight_results: Dict, load_dir, lm1=None, cache_file=None,
                      recompute: bool = False, progress_factory=None) -> Dict:
    """
    Compute tau, tau1, tau2 and taum for every path and store them in
    weight_results['path_data'] (keys 'tau', 'tau1', 'tau2', 'taum', 'has_tau').

    Merges the notebook's `load_path_order_data` and `compute_path_taus` into
    a single pass, so the order parameters of all paths are never held in
    memory at once. Per-path results are cached by path number in
    `cache_file` (.npz); only paths missing from the cache are read from
    <load_dir>/<pnr>/order.txt, so rerunning on a growing simulation only
    reads the new paths. The cache is discarded if the interfaces or lm1
    have changed.
    """
    from tistools import get_tau_staple, get_tau1_staple, get_tau2_staple

    D = weight_results['path_data']
    interfaces = weight_results['interfaces']
    pnrs = np.asarray(D['pnr']).ravel().astype(np.int64)
    n_paths = len(pnrs)
    start_intfs = np.asarray(D['start_intf']).astype(int)
    end_intfs = np.asarray(D['end_intf']).astype(int)

    # tau, tau1, tau2, has_tau per path number
    cache = {}
    if cache_file is not None and Path(cache_file).exists() and not recompute:
        try:
            c = np.load(cache_file)
            same_intf = np.allclose(c['interfaces'], interfaces)
            c_lm1 = float(c['lm1'])
            same_lm1 = (np.isnan(c_lm1) and lm1 is None) or (lm1 is not None and c_lm1 == lm1)
            if same_intf and same_lm1:
                cache = {int(p): (t, t1, t2, h) for p, t, t1, t2, h in
                         zip(c['pnr'], c['tau'], c['tau1'], c['tau2'], c['has_tau'])}
        except Exception:
            cache = {}

    tau = np.zeros(n_paths)
    tau1 = np.zeros(n_paths)
    tau2 = np.zeros(n_paths)
    has_tau = np.zeros(n_paths, dtype=bool)

    todo = []
    for i, pn in enumerate(pnrs):
        hit = cache.get(int(pn))
        if hit is None:
            todo.append(i)
        else:
            tau[i], tau1[i], tau2[i], has_tau[i] = hit

    n_missing = n_error = 0
    progress = progress_factory(pnrs[todo]) if (progress_factory and todo) else None
    for step, i in enumerate(todo):
        orders = load_order_from_path(Path(load_dir) / str(pnrs[i]))
        if orders is None:
            n_missing += 1
        else:
            start, end = start_intfs[i], end_intfs[i]
            try:
                tau[i] = get_tau_staple(orders, start, end, interfaces, lm1=lm1)
                tau1[i] = get_tau1_staple(orders, start, end, interfaces, lm1=lm1)
                tau2[i] = get_tau2_staple(orders, start, end, interfaces, lm1=lm1)
                has_tau[i] = True
            except Exception:
                n_error += 1
        if progress is not None:
            progress(step)

    if cache_file is not None and todo:
        for i in todo:
            cache[int(pnrs[i])] = (tau[i], tau1[i], tau2[i], has_tau[i])
        keys = np.fromiter(cache.keys(), dtype=np.int64)
        vals = np.array(list(cache.values()), dtype=float).reshape(-1, 4)
        np.savez(cache_file, pnr=keys, tau=vals[:, 0], tau1=vals[:, 1], tau2=vals[:, 2],
                 has_tau=vals[:, 3].astype(bool), interfaces=np.asarray(interfaces, dtype=float),
                 lm1=np.nan if lm1 is None else float(lm1))

    D['tau'] = tau
    D['tau1'] = tau1
    D['tau2'] = tau2
    D['taum'] = tau - tau1 - tau2
    D['has_tau'] = has_tau

    return {
        'n_paths': n_paths,
        'n_cached': n_paths - len(todo),
        'n_read': len(todo) - n_missing,
        'n_missing': n_missing,
        'n_error': n_error,
        'n_with_tau': int(has_tau.sum()),
    }


def compute_xi_running(D: Dict) -> np.ndarray:
    """
    Running estimate of the lm1 correction factor xi for every prefix of the paths.

    Ptype branch of the notebook's `compute_xi_from_trajs`: of the [0-] paths
    (weighted by their ensemble-0 occurrence path_f[:, 0]), xi is the fraction
    that ends on the right (LMR, RMR) rather than the left (LML, RML).
    Returns xi[n] for the first n + 1 paths (NaN until a [0-] path ends).
    """
    ptype = np.asarray(D['ptype']).astype(str)
    fac = np.asarray(D['path_f'])[:, 0].astype(float)
    ptype = np.where(ptype == 'L*L', 'LML', np.where(ptype == 'R*R', 'RMR', ptype))
    r_end = np.where(np.isin(ptype, ['LMR', 'RMR']), fac, 0.0)
    l_end = np.where(np.isin(ptype, ['LML', 'RML']), fac, 0.0)
    r_cum = np.cumsum(r_end)
    tot_cum = r_cum + np.cumsum(l_end)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot_cum > 0, r_cum / tot_cum, np.nan)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def error_analysis_staple(
    simdir: Annotated[str, typer.Argument(help="Path to the infretis simulation directory (containing infretis_data.txt and infretis.toml)")],
    data_file: Annotated[Optional[str], typer.Option("--data-file", help="Path to infretis_data.txt (default: <simdir>/infretis_data.txt)")] = None,
    toml_file: Annotated[Optional[str], typer.Option("--toml-file", help="Path to infretis.toml (default: <simdir>/infretis.toml)")] = None,
    nskip: Annotated[int, typer.Option("--nskip", help="Number of initial path entries to skip (default: 0)")] = 0,
    interval: Annotated[int, typer.Option("-i", "--interval", help="Interval between running estimates in paths (default: 1). Cost grows as ~n_paths^2/interval.")] = 1,
    pathlengths: Annotated[bool, typer.Option("--pathlengths", help="Also compute running flux/MFPT/rate estimates (balanced tau1/tau2) from the path lengths in <load-dir>/<pnr>/order.txt")] = False,
    load_dir: Annotated[Optional[str], typer.Option("--load-dir", help="Directory with one <pnr>/order.txt per path, used by --pathlengths (default: <simdir>/load)")] = None,
    time_unit: Annotated[float, typer.Option("--time-unit", help="Factor converting engine time units (timestep from infretis.toml) to the reported unit, e.g. 1e-12 for ps -> s (default: 1, report in engine time units)")] = 1.0,
    recompute_taus: Annotated[bool, typer.Option("--recompute-taus", help="Ignore the cached per-path taus (<outdir>/path_taus.npz) and reread every order.txt")] = False,
    outdir: Annotated[Optional[str], typer.Option("--outdir", help="Directory to save running estimates, block errors and plots in (default: <simdir>)")] = None,
    output: Annotated[Optional[str], typer.Option("-o", "--output", help="Output file for the report (default: stdout)")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Do not suppress the (very chatty) output of the weight/tistools routines")] = False,
    quiet_progress: Annotated[bool, typer.Option("-q", "--quiet", help="Do not show the progress bar")] = False,
):
    # The block-error writer saves figures; pick a non-interactive backend
    # before tistools pulls in pyplot, or this dies on a headless machine.
    import matplotlib
    matplotlib.use("Agg")

    try:
        # A directory containing an empty `tistools` namespace package lives
        # under inftools; strip it so `import tistools` resolves to the real,
        # pip-installed (editable) tistools package instead of that shadow.
        sys.path = [p for p in sys.path if 'inftools' not in p]
        from tistools import get_transition_probs_weights, construct_M_istar, global_pcross_msm_star, write_plot_block_error, write_running_estimates
        from tistools import mfpt_to_absorbing_staple, construct_tau_matrix_staple, mfpt_to_absorbing_staple_balanced, mfpt_istar, mfpt_istar_balanced
    except ImportError as e:
        print(f"Error: Could not import tistools: {e}", file=sys.stderr)
        print("Make sure tistools is installed or in your PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

    simdir = Path(simdir).resolve()
    if not simdir.exists():
        print(f"Error: Directory {simdir} does not exist", file=sys.stderr)
        sys.exit(1)

    outdir = Path(outdir).resolve() if outdir else simdir
    outdir.mkdir(parents=True, exist_ok=True)

    data_file = data_file or str(simdir / "infretis_data.txt")
    toml_file = toml_file or str(simdir / "infretis.toml")
    for f in (data_file, toml_file):
        if not Path(f).exists():
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    out = open(output, "w") if output else sys.stdout

    def log(msg=""):
        print(msg, file=out)

    log("=" * 80)
    log("RUNNING ESTIMATE + RECURSIVE BLOCK ERROR ANALYSIS (infretis STAPLE)")
    log("=" * 80)
    log(f"Simulation      : {simdir}")
    log(f"Data file       : {data_file}")
    log(f"TOML file       : {toml_file}")
    log(f"Output dir      : {outdir}")
    log(f"Interval        : {interval} paths")
    log(f"Skip from start : {nskip}")
    if pathlengths:
        load_dir = Path(load_dir).resolve() if load_dir else simdir / "load"
        if not load_dir.is_dir():
            print(f"Error: Directory {load_dir} does not exist (needed for --pathlengths)", file=sys.stderr)
            sys.exit(1)
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(toml_file, "rb") as ft:
            engine = tomllib.load(ft).get("engine", {})
        timestep = float(engine.get("timestep", 1.0))
        subcycles = int(engine.get("subcycles", 1))
        # tau is counted in stored frames, which are `subcycles` MD steps apart
        time_fac = timestep * subcycles * time_unit
        log(f"Load dir        : {load_dir}")
        log(f"Time per frame  : {timestep} x {subcycles} subcycles x {time_unit} = {time_fac:.6g}"
            + ("" if "timestep" in engine else "  (!! no [engine] timestep in toml, using 1)"))
    log()

    # -------------------------------------------------------------------------
    # 1. READING (infretis_data.txt + infretis.toml) + WEIGHTS
    # -------------------------------------------------------------------------
    print("Reading infretis data...", file=sys.stderr)
    t_load = time.time()
    with quiet(not verbose):
        weight_results = calculate_infretis_weights(data_file, toml_file, nskip=nskip)
        # Compute weight matrices, best with tr = False
        weight_matrices_results = compute_weight_matrices_weights(weight_results, tr=False)
    w_path = weight_matrices_results['weight_matrix_3d']
    w_path_2d = weight_matrices_results['weight_matrix_2d']
    lm1 = weight_results.get("lm1", None)

    D = weight_results['path_data']
    N_int = len(weight_results['interfaces'])
    N_paths = len(D['pnr'])
    log(f"Interfaces      : {N_int}  {weight_results['interfaces']}")
    log(f"lm1             : {lm1}")
    log(f"Paths available : {N_paths} (after skip)")

    path_w_full = D['path_w']
    path_f_full = D['path_f']
    is_ens0_full = (np.minimum(path_w_full, 1.0) != 0) & (path_f_full != 0)
    is_ens0_full = is_ens0_full[:, 0]   # True when path belongs to ensemble-0
    j_raw_full   = D["start_intf"].astype(np.intp)
    k_raw_full   = D["end_intf"].astype(np.intp)
    
    if pathlengths:
        if 'ptype' not in D:
            print("Error: --pathlengths needs ptype information in infretis_data.txt", file=sys.stderr)
            sys.exit(1)
        print(f"  loaded in {_fmt_time(time.time() - t_load)}", file=sys.stderr)
        print("Computing path lengths...", file=sys.stderr)
        t_tau = time.time()
        tau_factory = None if quiet_progress else (
            lambda pn: make_progress(pn, work=np.ones(len(pn)), unit="paths", label="pnr"))
        tau_info = compute_path_taus(weight_results, load_dir, lm1=lm1,
                                     cache_file=outdir / "path_taus.npz",
                                     recompute=recompute_taus, progress_factory=tau_factory)
        print(f"  path lengths done in {_fmt_time(time.time() - t_tau)} "
              f"({tau_info['n_cached']} cached, {tau_info['n_read']} read)", file=sys.stderr)
        log(f"Path lengths    : {tau_info['n_with_tau']}/{N_paths} paths "
            f"({tau_info['n_cached']} from cache, {tau_info['n_read']} read from order.txt)")
        if tau_info['n_missing'] or tau_info['n_error']:
            log(f"  !! {tau_info['n_missing']} paths without order.txt and {tau_info['n_error']} "
                f"failed tau computations; these keep their weight in P/Q but are left out of the tau averages")
        has_tau_full = D["has_tau"]
        tau_full     = D["tau"]
        tau1_full    = D["tau1"]
        tau2_full    = D["tau2"]
        taum_full    = D["taum"]
        # lm1 correction of the [0-] times: running xi from the ptype counts
        xi_full = compute_xi_running(D) if lm1 is not None else None

    # Define our snapshots based on interval
    start_cyc = max(10, interval)
    nskip_arr = np.arange(start_cyc, N_paths, interval, dtype=np.intp)
    n_snapshots = len(nskip_arr)
    if n_snapshots == 0:
        print(f"Error: only {N_paths} paths after skip; need more than {start_cyc} "
              f"for a single running estimate.", file=sys.stderr)
        sys.exit(1)

    # Q matrix is N_int x N_int. P matrix varies but we evaluate at n_int=N_int
    q_mat_stored = np.full((n_snapshots, N_int, N_int), np.nan)
    p_mat_stored = np.full((n_snapshots, N_int, N_int), np.nan)

    log(f"Running estimates: {n_snapshots} steps, {start_cyc} -> {N_paths} (step {interval})")

    # compute_rel_errors_2d sweeps block lengths up to n_estimates // 5, so
    # fewer than 5 estimates leaves nothing to analyse and fewer than ~25
    # leaves too few block lengths to show a plateau.
    if n_snapshots < 5:
        log()
        log(f"!! {n_snapshots} running estimates is below the minimum of 5 blocks: only the")
        log(f"!! running estimates will be written, NO block-error output. Use")
        log(f"!! --interval {max(N_paths // 25, 1)} or smaller.")
        print(f"WARNING: only {n_snapshots} estimates; no block-error output will be produced. "
              f"Try --interval {max(N_paths // 25, 1)}.", file=sys.stderr)
    elif n_snapshots < 25:
        log(f"  (note: only {n_snapshots} estimates; block lengths sweep to {n_snapshots // 5}, "
            f"which may be too few to show a plateau)")
    log()
    if pathlengths:
        print(f"  {n_snapshots} running-estimate steps to go", file=sys.stderr)
    else:
        print(f"  loaded in {_fmt_time(time.time() - t_load)}; "
              f"{n_snapshots} running-estimate steps to go", file=sys.stderr)
    
    # ── 2. Global norm_factor (unchanged across all subsets) ────
    path_w_c     = np.minimum(path_w_full, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_full = np.where(path_w_c != 0, path_f_full / path_w_c, 0.0)
    denom_full   = ratio_full.sum(axis=0)
    numer_full   = path_f_full.sum(axis=0)
    norm_full    = np.where(denom_full != 0, numer_full / denom_full, 0.0)
    weight_k_full = ratio_full * norm_full[np.newaxis, :]  # (N, n_ens)

    # Ensemble-0 total weight is CONSTANT regardless of subset size
    ens0_weight_full = float(weight_k_full[is_ens0_full, 0].sum())

    # Not-ensemble-0 mask
    not_ens0_full = ~is_ens0_full

    # ── 3. Helper: build wm3d + compute ploc + rates for one subset ─────
    RATE_LABELS = ["Flux", "Flux_nb", "MFPT_AB", "MFPT_nb_AB", "Rate", "Rate_nb"]

    def _rates_from_taus(M_mat, tsum3d, tw3d, pcross, xi):
        """
        Flux, MFPT(A->B) and rate from the running tau sums, balanced
        (tau1/tau2 pooled per turn point, weighted by the data behind each)
        and non-balanced (tau2 only), as in the notebook's MFPT/flux cells.
        """
        nan_out = np.full(len(RATE_LABELS), np.nan)
        if xi is not None and not (np.isfinite(xi) and xi > 0):
            return nan_out

        # (N+1) x N interface-space matrices: row 0 = [0-] (ensemble 0),
        # rows 1..N = start interface 0..N-1, summed over ensembles 1..N-1.
        tw_2d = np.zeros((N_int + 1, N_int))
        ts_2d = np.zeros((N_int + 1, N_int, 4))
        tw_2d[0, 0] = tw3d[0][0, 0]
        ts_2d[0, 0] = tsum3d[0][0, 0]
        for ens in range(1, N_int):
            tw_2d[1:] += tw3d[ens]
            ts_2d[1:] += tsum3d[ens]
        avg = np.zeros_like(ts_2d)
        nz = tw_2d > 0
        avg[nz] = ts_2d[nz] / tw_2d[nz][:, None]
        p_tau = {'tau': avg[..., 0], 'tau1': avg[..., 1], 'tau2': avg[..., 2],
                 'taum': avg[..., 3], 'weights': tw_2d}

        # lm1 correction: only part of the [0-] time is spent beyond lm1
        taum_xi = p_tau['taum'].copy()
        tau_0m = p_tau['tau'][0, 0]
        if xi is not None:
            taum_xi[0, 0] /= xi
            tau_0m /= xi

        NS = len(M_mat)
        N = NS // 2
        absor = [NS - 1]
        kept = list(range(NS - 1))
        try:
            # tau[0+], the excursion out of A (h1[0][0], absorbing in 0, 1 and NS-1)
            tau_0p = mfpt_istar_balanced(M_mat, p_tau)[2][0][0]
            tau_0p_nb = mfpt_istar(M_mat, p_tau)[2][0][0]
            # MFPT A -> B: absorbing in the last state, read off h2[0][0] (kept[0]
            # is state 0) with the dwell in A kept in (remove_initial_m=False)
            _, _, _, h2 = mfpt_to_absorbing_staple_balanced(
                M_mat, p_tau['tau1'], taum_xi, p_tau['tau2'], absor, kept,
                weights=tw_2d, remove_initial_m=False)
            _, _, _, h2_nb = mfpt_to_absorbing_staple(
                M_mat, construct_tau_matrix_staple(p_tau['tau1'], N),
                construct_tau_matrix_staple(taum_xi, N),
                construct_tau_matrix_staple(p_tau['tau2'], N),
                absor, kept, remove_initial_m=False)
        except (np.linalg.LinAlgError, ValueError):
            return nan_out

        with np.errstate(divide="ignore", invalid="ignore"):
            flux = 1.0 / ((tau_0m + tau_0p) * time_fac)
            flux_nb = 1.0 / ((tau_0m + tau_0p_nb) * time_fac)
        mfpt_AB = h2[0][0] * time_fac
        mfpt_nb_AB = h2_nb[0][0] * time_fac
        return np.array([flux, flux_nb, mfpt_AB, mfpt_nb_AB, flux * pcross, flux_nb * pcross])

    def _all_plocs_and_rates_from_prefix(n_rows: int):
        plocs_out = np.ones(N_int)

        j_raw_sub  = j_raw_full[:n_rows]
        k_raw_sub  = k_raw_full[:n_rows]
        wk_sub     = weight_k_full[:n_rows]
        not_e0_sub = not_ens0_full[:n_rows]
        is_e0_sub  = is_ens0_full[:n_rows]

        if pathlengths:
            # Weighted sums of [tau, tau1, tau2, taum] per ensemble and
            # (start, end), plus the weight behind them. Only paths with a
            # computed tau enter, so a missing order.txt does not drag the
            # averages towards zero.
            taus_sub = np.stack([tau_full[:n_rows], tau1_full[:n_rows],
                                 tau2_full[:n_rows], taum_full[:n_rows]], axis=1)
            ht_sub   = has_tau_full[:n_rows]
            tsum3d   = {ens: np.zeros((N_int, N_int, 4)) for ens in range(N_int)}
            tw3d     = {ens: np.zeros((N_int, N_int)) for ens in range(N_int)}
            m_e0 = is_e0_sub & ht_sub
            w_e0 = wk_sub[m_e0, 0]
            tsum3d[0][0, 0] = w_e0 @ taus_sub[m_e0]
            tw3d[0][0, 0] = w_e0.sum()

        ens0_w = float(wk_sub[is_e0_sub, 0].sum())

        for n_int in range(2, N_int + 1):
            L = n_int - 1

            j_out = (j_raw_sub < 0) | (j_raw_sub >= n_int)
            k_out = (k_raw_sub < 0) | (k_raw_sub >= n_int)
            skip  = (j_out & k_out) | ((j_raw_sub >= L) & (k_raw_sub >= L))
            valid = ~skip

            # ── clip INSIDE the loop to the current n_int size ──
            j_clipped = np.clip(j_raw_sub, 0, n_int - 1)
            k_clipped = np.clip(k_raw_sub, 0, n_int - 1)

            wm3d = {ens: np.zeros((n_int, n_int)) for ens in range(n_int)}
            wm3d[0][0, 0] = ens0_w

            for ens in range(1, n_int):
                w_ens = wk_sub[:, ens]
                mask  = valid & not_e0_sub & (w_ens != 0)
                if not np.any(mask):
                    continue
                jm = j_clipped[mask]
                km = k_clipped[mask]
                wm = w_ens[mask]

                # Self-transitions only count at 0 -> [0, 0] and L -> [L, L-1]
                self_m = jm == km
                m_0 = self_m & (jm == 0)
                m_L = self_m & (jm == L)
                wm3d[ens][0, 0] += wm[m_0].sum()
                wm3d[ens][L, L - 1] += wm[m_L].sum()
                off = ~self_m
                np.add.at(wm3d[ens], (jm[off], km[off]), wm[off])

                if pathlengths and n_int == N_int:
                    # Same cells as wm3d, restricted to paths with a tau
                    keep = (off | m_0 | m_L) & ht_sub[mask]
                    kt = np.where(m_L, L - 1, km)[keep]
                    jt = jm[keep]
                    wt = wm[keep]
                    np.add.at(tw3d[ens], (jt, kt), wt)
                    np.add.at(tsum3d[ens], (jt, kt), wt[:, None] * taus_sub[mask][keep])

            p_mat, q_mat = get_transition_probs_weights(wm3d)
            M_mat    = construct_M_istar(p_mat, max(4, 2 * n_int), n_int)
            try:
                _, _, y1, _ = global_pcross_msm_star(M_mat)
            except Exception:
                y1 = [[np.nan]]
            plocs_out[L] = float(y1[0][0])

        # p_mat, q_mat and M_mat now belong to the full n_int == N_int model
        if not pathlengths:
            return plocs_out, p_mat, q_mat, np.full(len(RATE_LABELS), np.nan)
        xi = None if xi_full is None else xi_full[n_rows - 1]
        return plocs_out, p_mat, q_mat, _rates_from_taus(M_mat, tsum3d, tw3d, plocs_out[-1], xi)

    # -------------------------------------------------------------------------
    # 2. RUNNING AVERAGE LOOP
    # -------------------------------------------------------------------------
    ploc_MSM_stored = np.full((n_snapshots, N_int), np.nan)
    rate_stored = np.full((n_snapshots, len(RATE_LABELS)), np.nan)
    cycles = []

    progress = None if quiet_progress else make_progress(nskip_arr)

    t0 = time.time()
    with quiet(not verbose):
        for snap_i, n_rows in enumerate(nskip_arr):
            ploc_MSM_stored[snap_i, :], p_mat_stored[snap_i, :, :], q_mat_stored[snap_i, :, :], rate_stored[snap_i, :] = _all_plocs_and_rates_from_prefix(int(n_rows))
            cycles.append(n_rows)
            if progress is not None:
                progress(snap_i)
    log(f"Running estimates completed in {_fmt_time(time.time() - t0)}.")

    runav_files = {
        "pcross_runav.txt": (ploc_MSM_stored, "Pcross"),
        "qmat_runav.txt": (q_mat_stored.reshape(n_snapshots, -1), "Qmat"),
        "pmat_runav.txt": (p_mat_stored.reshape(n_snapshots, -1), "Pmat"),
    }
    # write_running_estimates dumps every array to stdout as a debug print
    with quiet(not verbose):
        for fname, (data, label) in runav_files.items():
            write_running_estimates(outdir / fname, cycles, data, label)
        if pathlengths:
            rate_cols = [x for i, lab in enumerate(RATE_LABELS) for x in (rate_stored[:, i], lab)]
            write_running_estimates(outdir / "rate_runav.txt", cycles, *rate_cols)
    written = list(runav_files) + (["rate_runav.txt"] if pathlengths else [])

    # -------------------------------------------------------------------------
    # 3. VECTORIZED BLOCK ERROR ANALYSIS
    # -------------------------------------------------------------------------
    # Trim nans if any
    valid_rows = ~np.any(np.isnan(ploc_MSM_stored), axis=1)
    
    # We skip early transients (e.g., first 5) for stable error bounds
    trim_start = min(5, len(valid_rows) // 1000)
    
    runav_pcross = ploc_MSM_stored[valid_rows][trim_start:]
    runav_qmat   = q_mat_stored[valid_rows][trim_start:]
    runav_pmat   = p_mat_stored[valid_rows][trim_start:]
    # Rates are NaN until every piece they need has been sampled (e.g. no
    # [0-] path has ended yet for xi), so they get their own valid rows
    rate_rows    = valid_rows & np.all(np.isfinite(rate_stored), axis=1)
    runav_rate   = rate_stored[rate_rows][trim_start:]
    
    maxbll = len(runav_pcross) // 5
    sizes  = np.arange(1, maxbll + 1, dtype=np.intp)
    maxbll_rate = len(runav_rate) // 5
    sizes_rate  = np.arange(1, maxbll_rate + 1, dtype=np.intp)

    if maxbll >= 1:
        print("Computing block errors...", file=sys.stderr)
        t_block = time.time()
        log()
        log(f"Using {len(valid_rows) - trim_start} valid snapshots for error analysis (skipping first {trim_start} for stability)")

        block_sets = [
            ("pcross", runav_pcross),
            ("qmat", runav_qmat.reshape(len(runav_qmat), -1)),
            ("pmat", runav_pmat.reshape(len(runav_pmat), -1)),
        ]
        do_rates = pathlengths and maxbll_rate >= 1
        if do_rates:
            block_sets.append(("rate", runav_rate))
            log(f"Using {len(runav_rate)} snapshots with finite rates for the rate error analysis")
        elif pathlengths:
            log(f"!! Only {len(runav_rate)} snapshots with finite rates: skipping rate block errors")

        errs = {}
        with quiet(not verbose):
            for name, runav in block_sets:
                errs[name] = compute_rel_errors_2d(runav, sizes_rate if name == "rate" else sizes)
                write_plot_block_error(str(outdir / f"{name}_block_errors_{interval}"), runav, errs[name], interval)
                written.append(f"{name}_block_errors_{interval}.txt / .png")
        print(f"  block errors done in {_fmt_time(time.time() - t_block)}", file=sys.stderr)

        err_pcross = errs["pcross"]
        err_qmat = errs["qmat"]
        err_pmat = errs["pmat"]

        # ---------------------------------------------------------------------
        # 4. SUMMARY OUTPUT
        # ---------------------------------------------------------------------
        plateau_mask = sizes > maxbll // 2
        
        # P_cross final interface analysis
        best_pcross = runav_pcross[-1, -1]
        rel_err_pcross = err_pcross[:, -1]
        half_av_err = rel_err_pcross[plateau_mask].mean() if plateau_mask.any() else rel_err_pcross[-1]
        Nstat_ineff = (half_av_err / rel_err_pcross[0])**2 if rel_err_pcross[0] != 0 else 0.0

        # Q / P Matrix average errors across all non-zero elements
        avg_qmat_err = np.nanmean(err_qmat[-1, :]) 
        avg_pmat_err = np.nanmean(err_pmat[-1, :])

        summary = f"""
    Block Error Summary:
    ---------------------------------------------------
    Data points analyzed          : {len(runav_pcross)}
    Max block length              : {maxbll}
    
    Final P_cross                 : {best_pcross:.6g}
    P_cross Rel. Error (Plateau)  : {half_av_err:.4f} ({half_av_err*100:.2f}%)
    P_cross Stat. Inefficiency    : {Nstat_ineff:.1f}
    
    Q Matrix Average Rel Error    : {avg_qmat_err:.4f}
    P Matrix Average Rel Error    : {avg_pmat_err:.4f}
    
    """
        if do_rates:
            # Per quantity: final running estimate and plateau relative error
            err_rate = errs["rate"]
            plateau_rate = sizes_rate > maxbll_rate // 2
            rel_rate = (err_rate[plateau_rate].mean(axis=0) if plateau_rate.any()
                        else err_rate[-1])
            best_rate = runav_rate[-1]
            unit = "engine time units" if time_unit == 1.0 else f"engine time units x {time_unit:g}"
            rows = [f"    {lab:<12}: {val:>13.6e}  +- {val * rel:.3e}  ({rel * 100:.2f}%)"
                    for lab, val, rel in zip(RATE_LABELS, best_rate, rel_rate)]
            summary += (
                f"\n    Rates (time in {unit}; _nb = tau2 only, not balanced):\n"
                f"    ---------------------------------------------------\n"
                + "\n".join(rows)
                + f"\n    1/MFPT_AB    : {1 / best_rate[2]:>13.6e}  (should match Rate)"
                + (f"\n    xi (lm1)     : {xi_full[nskip_arr[-1] - 1]:.6g}" if xi_full is not None else "")
                + "\n"
            )
        log(summary)
        # The report went to a file: still show the summary on the terminal.
        if output:
            print(summary)
    else:
        log()
        log(f"!! Only {len(runav_pcross)} valid running estimates: skipping block error analysis.")

    log(f"Written to {outdir}:")
    for fname in written:
        log(f"  {fname}")
    log()
    log("Read the relative error against block length and take the plateau: short")
    log("blocks are still correlated and underestimate the error, while very long")
    log("blocks leave too few of them for the standard error to be meaningful.")
    log("=" * 80)

    if output:
        out.close()
        print(f"Report written to {output}", file=sys.stderr)
