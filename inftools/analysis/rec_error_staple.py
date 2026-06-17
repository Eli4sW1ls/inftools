#!/usr/bin/env python3
"""
tistools-running-error-staple-opt: Optimized running estimate error analysis for APPTIS (STAPLE).

Computes running estimates at regular intervals using vectorized recursive formulas.
Outputs:
  - Running estimates of P_cross, Q matrix elements, and P (MSM) matrix elements.
  - Highly optimized block error analysis (numpy reshape, no Python per-element loops).

Usage:
    python tistools-running-error-staple-opt.py <simulation_dir> [options]
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
from typing import Annotated, Dict, Optional

import typer

# Adjust path if needed to find tistools
sys.path = [p for p in sys.path if 'inftools' not in p]
sys.path.insert(0, os.path.abspath('..'))


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
    if lm1 is not None:
        print(f"read lm1 from toml: {lm1}")

    
    # Load data
    data = np.loadtxt(data_file, dtype=str, usecols=np.arange(37))
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
                                print("nooooo")
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
# CLI PARSER 
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Running estimate error analysis for APPTIS (STAPLE) simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("simdir", type=str, help="Path to REPPTIS simulation directory")
    parser.add_argument("--interval", "-i", type=int, default=10000, help="Interval between running estimates in cycles (default: 10000)")
    parser.add_argument("--skip", "-s", type=int, default=0, help="Cycles to skip from beginning (default: 0)")
    parser.add_argument("--pathlengths", action="store_true", help="Include path length information in output")
    parser.add_argument("--load-orders", action="store_true", help="Load order parameters from .npy files")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file for results (default: stdout)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print final summary")
    return parser.parse_args()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def error_analysis_staple(
    simdir: Annotated[str, typer.Argument(help="Path to simulation directory")],
    interval: Annotated[int, typer.Option("-i", "--interval", help="Interval between running estimates in cycles (default: 1)")] = 1,
    skip: Annotated[int, typer.Option("-s", "--skip", help="Cycles to skip from beginning (default: 0)")] = 0,
    pathlengths: Annotated[bool, typer.Option("--pathlengths", help="Include path length information in output")] = False,
    load_orders: Annotated[bool, typer.Option("--load-orders", help="Load order parameters from .npy files")] = False,
    output: Annotated[Optional[str], typer.Option("-o", "--output", help="Output file for results (default: stdout)")] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Only print final summary")] = False,
):
    # args = parse_args()

    # Dynamic imports for PyRETIS/tistools topology
    try:
        from tistools import get_transition_probs_weights, construct_M_istar, global_pcross_msm_star, write_plot_block_error, write_running_estimates
        from tistools import mfpt_to_absorbing_staple, construct_tau_matrix_staple, mfpt_to_absorbing_staple_balanced, mfpt_istar, mfpt_istar_balanced
        # If your notebook used specific tistools extraction functions, import them here
    except ImportError as e:
        print(f"Error: Could not import necessary tistools functions: {e}", file=sys.stderr)
        sys.exit(1)

    simdir = Path(simdir).resolve()
    if not simdir.exists():
        print(f"Error: Directory {simdir} does not exist", file=sys.stderr)
        sys.exit(1)

    out = open(output, "w") if output else sys.stdout

    def log(msg):
        if not quiet:
            print(msg, file=out)

    log("=" * 80)
    log("OPTIMIZED RUNNING ESTIMATE ERROR ANALYSIS")
    log("=" * 80)
    log(f"Simulation: {simdir}")
    log(f"Interval: {interval} cycles")
    log(f"Skip from start: {skip}\n")

    # 1. Load Data [cite: 6]
    # Calculate infretis weights
    log("=== STEP 1: CALCULATING INFRETIS WEIGHTS ===")
    weight_results = calculate_infretis_weights(simdir / "infretis_data.txt", simdir / "infretis.toml", nskip=skip)
    lm1 = weight_results.get("lm1", None)
    log(f"lm1 value from weights: {lm1}")

    log("\nWeight calculation summary:")
    log(f"Number of interfaces: {len(weight_results['interfaces'])}")
    log(f"Interfaces: {weight_results['interfaces']}")
    log(f"Number of paths processed: {len(weight_results['path_data']['pnr'])}")

    D = weight_results['path_data']
    N_int = len(weight_results['interfaces'])
    N_paths = len(D['pnr'])
    path_w_full = D['path_w']
    path_f_full = D['path_f']
    is_ens0_full = (np.minimum(path_w_full, 1.0) != 0) & (path_f_full != 0)
    is_ens0_full = is_ens0_full[:, 0]   # True when path belongs to ensemble-0
    j_raw_full   = D["start_intf"].astype(np.intp)
    k_raw_full   = D["end_intf"].astype(np.intp)
    
    if pathlengths:
        # TODO
        tau_full     = D["tau"]
        tau1_full    = D["tau1"]
        tau2_full    = D["tau2"]
        taum_full    = D["taum"]
    
    # Compute weight matrices, best with tr = False
    log("\n=== STEP 2: COMPUTING WEIGHT MATRICES ===")
    weight_matrices_results = compute_weight_matrices_weights(weight_results, tr=False)
    w_path = weight_matrices_results['weight_matrix_3d']
    w_path_2d = weight_matrices_results['weight_matrix_2d']
    log("Weight matrices computed successfully!")
    
    # Define our snapshots based on interval
    start_cyc = max(10, interval)
    nskip_arr = np.arange(start_cyc, N_paths, interval, dtype=np.intp)
    n_snapshots = len(nskip_arr)

    # Output storage
    # P_cross storage (just final interface or all L interfaces)
    ploc_MSM_stored = np.full((n_snapshots, N_int), np.nan)
    
    # Store flattened Q (p_mat) and P (M_mat) matrices
    # Q matrix is N_int x N_int. P matrix varies but we evaluate at n_int=N_int
    q_mat_stored = np.full((n_snapshots, N_int, N_int), np.nan)
    p_mat_stored = np.full((n_snapshots, N_int, N_int), np.nan)

    log(f"Computing running averages: {start_cyc} → {N_paths} (step {interval})")
    log(f"Total snapshots: {n_snapshots}  |  Interfaces: {N_int}")
    
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
    def _all_plocs_and_rates_from_prefix(n_rows: int):
        plocs_out = np.ones(N_int)

        j_raw_sub  = j_raw_full[:n_rows]
        k_raw_sub  = k_raw_full[:n_rows]
        wk_sub     = weight_k_full[:n_rows]
        if pathlengths:
            tau_sub    = tau_full[:n_rows]
            tau1_sub   = tau1_full[:n_rows]
            tau2_sub   = tau2_full[:n_rows]
            taum_sub   = taum_full[:n_rows]

        not_e0_sub = not_ens0_full[:n_rows]
        is_e0_sub  = is_ens0_full[:n_rows]
        
        # We will also compute rate/flux using the global N_int matrix
        t3d  = {ens: np.zeros((N_int, N_int)) for ens in range(N_int)}
        t1_3d = {ens: np.zeros((N_int, N_int)) for ens in range(N_int)}
        t2_3d = {ens: np.zeros((N_int, N_int)) for ens in range(N_int)}
        tm_3d = {ens: np.zeros((N_int, N_int)) for ens in range(N_int)}

        ens0_w = float(wk_sub[is_e0_sub, 0].sum())
        
        # Compute tau expectations for ensemble 0
        if pathlengths:
                mask_e0 = is_e0_sub
                wm_e0 = wk_sub[mask_e0, 0]
                t3d[0][0, 0] = (wm_e0 * tau_sub[mask_e0]).sum()
                t1_3d[0][0, 0] = (wm_e0 * tau1_sub[mask_e0]).sum()
                t2_3d[0][0, 0] = (wm_e0 * tau2_sub[mask_e0]).sum()
                tm_3d[0][0, 0] = (wm_e0 * taum_sub[mask_e0]).sum()

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
                
                if pathlengths:
                    t_m = tau_sub[mask]
                    t1_m = tau1_sub[mask]
                    t2_m = tau2_sub[mask]
                    tm_m = taum_sub[mask]

                self_m = jm == km
                if np.any(self_m):
                    m_0 = self_m & (jm == 0)
                    wm3d[ens][0, 0] += wm[m_0].sum()
                    if n_int == N_int and pathlengths:
                        t3d[ens][0, 0] += (wm[m_0] * t_m[m_0]).sum()
                        t1_3d[ens][0, 0] += (wm[m_0] * t1_m[m_0]).sum()
                        t2_3d[ens][0, 0] += (wm[m_0] * t2_m[m_0]).sum()
                        tm_3d[ens][0, 0] += (wm[m_0] * tm_m[m_0]).sum()
                        
                    if L > 0:
                        mL = self_m & (jm == L)
                        wm3d[ens][L, L - 1] += wm[mL].sum()
                        if n_int == N_int and pathlengths:
                            t3d[ens][L, L - 1] += (wm[mL] * t_m[mL]).sum()
                            t1_3d[ens][L, L - 1] += (wm[mL] * t1_m[mL]).sum()
                            t2_3d[ens][L, L - 1] += (wm[mL] * t2_m[mL]).sum()
                            tm_3d[ens][L, L - 1] += (wm[mL] * tm_m[mL]).sum()

                off = ~self_m
                if np.any(off):
                    np.add.at(wm3d[ens], (jm[off], km[off]), wm[off])
                    if n_int == N_int and pathlengths:
                        np.add.at(t3d[ens], (jm[off], km[off]), wm[off] * t_m[off])
                        np.add.at(t1_3d[ens], (jm[off], km[off]), wm[off] * t1_m[off])
                        np.add.at(t2_3d[ens], (jm[off], km[off]), wm[off] * t2_m[off])
                        np.add.at(tm_3d[ens], (jm[off], km[off]), wm[off] * tm_m[off])

            p_mat, q_mat = get_transition_probs_weights(wm3d)
            M_mat    = construct_M_istar(p_mat, max(4, 2 * n_int), n_int)
            _, _, y1, _ = global_pcross_msm_star(M_mat)
            plocs_out[L] = float(y1[0][0])
            
            if n_int == N_int:
                # We also compute the rates using the N_int dimensional M_mat & wm3d
                M_mat_N = M_mat

        # Now compute 2D tau arrays
        if not pathlengths:
            return plocs_out, p_mat, q_mat, (None, None, None, None, None, None)
        
        ws_2d = np.zeros((N_int + 1, N_int))
        ts_2d = np.zeros((N_int + 1, N_int))
        t1s_2d = np.zeros((N_int + 1, N_int))
        t2s_2d = np.zeros((N_int + 1, N_int))
        tms_2d = np.zeros((N_int + 1, N_int))
        
        ws_2d[0, 0] = wm3d[0][0, 0]
        ts_2d[0, 0] = t3d[0][0, 0]
        t1s_2d[0, 0] = t1_3d[0][0, 0]
        t2s_2d[0, 0] = t2_3d[0][0, 0]
        tms_2d[0, 0] = tm_3d[0][0, 0]
        
        for i in range(1, N_int):
            ws_2d[1:, :] += wm3d[i]
            ts_2d[1:, :] += t3d[i]
            t1s_2d[1:, :] += t1_3d[i]
            t2s_2d[1:, :] += t2_3d[i]
            tms_2d[1:, :] += tm_3d[i]
            
        mask = ws_2d > 0
        t_avg = np.zeros_like(ws_2d)
        t1_avg = np.zeros_like(ws_2d)
        t2_avg = np.zeros_like(ws_2d)
        tm_avg = np.zeros_like(ws_2d)
        
        t_avg[mask] = ts_2d[mask] / ws_2d[mask]
        t1_avg[mask] = t1s_2d[mask] / ws_2d[mask]
        t2_avg[mask] = t2s_2d[mask] / ws_2d[mask]
        tm_avg[mask] = tms_2d[mask] / ws_2d[mask]
        
        p_tau = {
            'tau': t_avg, 'tau1': t1_avg, 'tau2': t2_avg, 'taum': tm_avg
        }
        
        #TODO
        dt = 1.0  # Time step in ps (adjust as needed)
        subc = 1.0  # Subcycles (adjust as needed)
        absor = [N_int - 1]  # Absorbing states
        kept = list(range(N_int-1))  # All states kept
        xi_val = 1
        
        mfpt_bal = mfpt_istar_balanced(M_mat_N, p_tau)
        mfpt_0 = mfpt_bal[2][0][0]
        mfpt_nb_obj = mfpt_istar(M_mat_N, p_tau)
        mfpt_nb_0 = mfpt_nb_obj[2][0][0]
        
        p_taumm = p_tau['taum'].copy()
        if xi_val is not None:
            p_taumm[0,0] /= xi_val
        t1m = construct_tau_matrix_staple(p_tau['tau1'], N_int)
        tmm = construct_tau_matrix_staple(p_taumm, N_int)
        t2m = construct_tau_matrix_staple(p_tau['tau2'], N_int)
        
        _, _, h1mfpt, _ = mfpt_to_absorbing_staple_balanced(M_mat_N, p_tau['tau1'], p_taumm, p_tau['tau2'], absor, kept)
        _, _, h1mfpt_nb, _ = mfpt_to_absorbing_staple(M_mat_N, t1m, tmm, t2m, absor, kept)
        
        mfpt_AB = h1mfpt[0][0] * dt * subc * 1e-12
        mfpt_nb_AB = h1mfpt_nb[0][0] * dt * subc * 1e-12
        
        tau_z0 = p_tau['tau'][0,0]/xi_val if xi_val is not None else p_tau['tau'][0,0]
        flux = 1 / ((tau_z0 + mfpt_0) * dt * subc * 1e-12)
        flux_nb = 1 / ((tau_z0 + mfpt_nb_0) * dt * subc * 1e-12)
        
        rate = flux * plocs_out[-1]
        rate_nb = flux_nb * plocs_out[-1]
        print(f"rate = {rate:.6e}  |  rate_nb = {rate_nb:.6e}")
        
        return plocs_out, p_mat, q_mat, (flux, flux_nb, mfpt_AB, mfpt_nb_AB, rate, rate_nb)

    # -------------------------------------------------------------------------
    # 2. RUNNING AVERAGE LOOP
    # -------------------------------------------------------------------------
    checkpoint_every = 100
    checkpoint_path = simdir / "ploc_MSM_stored.npy"
    
    ploc_MSM_stored = np.full((n_snapshots, N_int), np.nan)
    rate_stored = np.full((n_snapshots, 6), np.nan)
    cycles = []
    
    for snap_i, n_rows in enumerate(nskip_arr):        
        ploc_MSM_stored[snap_i, :], p_mat_stored[snap_i, :, :], q_mat_stored[snap_i, :, :], rate_stored[snap_i, :] = _all_plocs_and_rates_from_prefix(int(n_rows))
        
        # if snap_i % checkpoint_every == 0 or snap_i == n_snapshots - 1:
        #     np.save(checkpoint_path, ploc_MSM_stored)
        #     if snap_i % 500 == 0:
        #         vals = ploc_MSM_stored[snap_i]
        #         print(f"  snap {snap_i:4d}/{n_snapshots}  "
        #                 f"nskip={n_rows:6d}  plocs={np.array2string(vals, precision=4)}")
        #         print(f"                                rate={rate_stored[snap_i, 4]:.6e}")
        cycles.append(n_rows)
            
    write_running_estimates(simdir / "pcross_runav.txt", cycles, ploc_MSM_stored, "Pcross")
    write_running_estimates(simdir / "qmat_runav.txt", cycles, q_mat_stored.reshape(n_snapshots, -1), "Qmat")
    write_running_estimates(simdir / "pmat_runav.txt", cycles, p_mat_stored.reshape(n_snapshots, -1), "Pmat")
    write_running_estimates(simdir / "rate_runav.txt", cycles, rate_stored[:, 0], "Flux", rate_stored[:, 1], "Flux_nb", rate_stored[:, 2], "MFPT_AB", rate_stored[:, 3], "MFPT_nb_AB", rate_stored[:, 4], "Rate", rate_stored[:, 5], "Rate_nb")
    print(f"✓ Done. ploc Shape: {ploc_MSM_stored.shape}")

    # -------------------------------------------------------------------------
    # 3. VECTORIZED BLOCK ERROR ANALYSIS
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("COMPUTING BLOCK ERRORS")
    log("=" * 80)

    # Trim nans if any
    valid_rows = ~np.any(np.isnan(ploc_MSM_stored), axis=1)
    
    # We skip early transients (e.g., first 5) for stable error bounds
    trim_start = min(5, len(valid_rows) // 1000)
    log(f"Using {len(valid_rows) - trim_start} valid snapshots for error analysis (skipping first {trim_start} for stability)")    
    
    runav_pcross = ploc_MSM_stored[valid_rows][trim_start:]
    runav_qmat   = q_mat_stored[valid_rows][trim_start:]
    runav_pmat   = p_mat_stored[valid_rows][trim_start:]
    runav_rate   = rate_stored[valid_rows][trim_start:]
    
    maxbll = len(runav_pcross) // 5
    sizes  = np.arange(1, maxbll + 1, dtype=np.intp)

    # Compute errors for P_cross (all L interfaces at once)
    log(f"-> Computing P_cross errors...")
    err_pcross = compute_rel_errors_2d(runav_pcross, sizes)
    write_plot_block_error(simdir / f"pcross_block_errors_{interval}.txt", runav_pcross, err_pcross, interval)
    
    # Compute errors for Q matrix (all N_int x N_int elements at once)
    log(f"-> Computing Q matrix errors (shape: {N_int}x{N_int})...")
    err_qmat = compute_rel_errors_2d(runav_qmat.reshape(len(runav_qmat), -1), sizes)
    write_plot_block_error(simdir / f"qmat_block_errors_{interval}.txt", runav_qmat, err_qmat, interval)
    
    # Compute errors for P matrix (MSM)
    log(f"-> Computing P (MSM) matrix errors (shape: {N_int}x{N_int})...")
    err_pmat = compute_rel_errors_2d(runav_pmat.reshape(len(runav_pmat), -1), sizes)
    write_plot_block_error(simdir / f"pmat_block_errors_{interval}.txt", runav_pmat, err_pmat, interval)
    
    # Compute errors for rates (flux, MFPT, etc.)
    log(f"-> Computing rate errors...")
    err_rate = compute_rel_errors_2d(runav_rate, sizes)
    write_plot_block_error(simdir / f"rate_block_errors_{interval}.txt", runav_rate, err_rate, interval)

    # -------------------------------------------------------------------------
    # 4. SUMMARY OUTPUT
    # -------------------------------------------------------------------------
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
    
    Rate Average Rel Error           : {np.nanmean(err_rate[-1, :]):.4f}
    """
    
    # Only print standard output if not quiet, but always print summary
    if out != sys.stdout:
        print(summary)
    log(summary)

    if output:
        out.close()
        print(f"Results written to {output}")

# if __name__ == "__main__":
#     main()