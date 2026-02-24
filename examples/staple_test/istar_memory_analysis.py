import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def memory_analysis(w_path, tr=False):
    """
    Analyze memory effects in transition paths by calculating conditional crossing probabilities.
    """
    n_int = list(w_path.values())[0].shape[0]
    q_k = np.zeros([2, n_int-1, n_int, n_int])
    for ens in range(1, n_int):
        if ens not in w_path: continue 
        w_ens = w_path[ens].copy()
        if tr:
            w_ens += w_ens.T
        for i in range(w_ens.shape[0]):
            for k in range(w_ens.shape[0]):
                counts = np.zeros(2)
                if i == k:
                    if i == 0:
                        q_k[0][ens-1][i][k] = 1
                        continue
                    else:
                        continue
                elif i == 0 and k == 1 and ens == 1:
                    q_k[0][ens-1][i][k] = (np.sum(w_ens[i][k:])) / (np.sum(w_ens[i][k-1:]))
                    q_k[1][ens-1][i][k] = np.sum(w_ens[i][k-1:])
                    continue
                elif i < k:
                    if i <= ens <= k:
                        counts += [np.sum(w_ens[i][k:]), np.sum(w_ens[i][k-1:])]
                elif i > k:
                    if k+2 <= ens <= i+1:
                        counts += [np.sum(w_ens[i][:k+1]), np.sum(w_ens[i][:k+2])]

                q_k[0][ens-1][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
                q_k[1][ens-1][i][k] = counts[1]
        
    q_tot = np.ones([2, n_int, n_int])
    for i in range(n_int):
        for k in range(n_int):
            counts = np.zeros(2)
            if i == k:
                if i == 0:
                    q_tot[0][i][k] = 1
                    continue
                else:
                    q_tot[0][i][k] = 0
                    continue
            elif i == 0 and k == 1:
                # Handle ensemble 1 separately if possible, or assume typical structure
                if 1 in w_path:
                    q_tot[0][i][k] = (np.sum(w_path[1][i][k:])) / (np.sum(w_path[1][i][k-1:]))
                    q_tot[1][i][k] = np.sum(w_path[1][i][k-1:])
                continue
            elif i < k:
                for pe_i in range(i+1, k+1):
                    if pe_i > n_int-1:
                        break
                    if pe_i in w_path:
                        counts += [np.sum(w_path[pe_i][i][k:]), np.sum(w_path[pe_i][i][k-1:])]
            elif i > k:
                for pe_i in range(k+2, i+2):
                    if pe_i > n_int-1:
                        break
                    if pe_i in w_path:
                        counts += [np.sum(w_path[pe_i][i][:k+1]), np.sum(w_path[pe_i][i][:k+2])]

            q_tot[0][i][k] = counts[0] / counts[1] if counts[1] > 0 else np.nan
            q_tot[1][i][k] = counts[1]


    return q_k, q_tot

def generate_state_labels(n_interfaces):
    state_labels = []
    middle = n_interfaces 
    
    for i in range(2*n_interfaces):
        if i == 0:
            state_labels.append("[0$^-]")
        elif i == 1:
            state_labels.append("[0←]")
        elif i == 2:
            state_labels.append("[0→]")
        elif i <= middle:
            state_labels.append(f"[{i-2}$\\subset$]")
        elif middle < i < 2*n_interfaces - 1:
            state_labels.append(f"[{i-middle}$\\supset$]")
        else:
            state_labels.append(f"[{i-middle}]")
    return state_labels

def calculate_memory_effect_index(q_probs, q_weights, q_errors=None, min_samples=5):
    n_interfaces = q_probs.shape[0]
    
    forward_variation = np.zeros(n_interfaces)
    forward_variation.fill(np.nan)
    forward_variation_error = np.zeros(n_interfaces)
    forward_variation_error.fill(np.nan)              
    forward_sample_sizes = np.zeros(n_interfaces, dtype=int)
    
    backward_variation = np.zeros(n_interfaces)
    backward_variation.fill(np.nan)
    backward_variation_error = np.zeros(n_interfaces)  
    backward_variation_error.fill(np.nan)              
    backward_sample_sizes = np.zeros(n_interfaces, dtype=int)
    
    for k in range(1, n_interfaces):
        q_values = []
        weights = []
        q_errors_k = []

        for i in range(max(1, k-1)): 
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples:
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
                if q_errors is not None and not np.isnan(q_errors[i, k]):
                    q_errors_k.append(q_errors[i, k])
                else:
                    binomial_error = np.sqrt(q_probs[i, k] * (1 - q_probs[i, k]) / q_weights[i, k]) if q_weights[i, k] > 0 else 0
                    q_errors_k.append(binomial_error)
        
        if len(q_values) >= 2:
            q_values = np.array(q_values)
            weights = np.array(weights)
            q_errors_arr = np.array(q_errors_k)
            total_samples = np.sum(weights)
            forward_sample_sizes[k] = total_samples
            std_dev = np.std(q_values)
            forward_variation[k] = std_dev * 100
            
            n = len(q_values)
            mean_q = np.mean(q_values)
            if std_dev > 0 and not np.any(np.isnan(q_errors_arr)):
                partial_derivs = (q_values - mean_q) / ((n - 1) * std_dev)
                std_error_squared = np.sum((partial_derivs * q_errors_arr) ** 2)
                std_error = np.sqrt(std_error_squared)
                forward_variation_error[k] = std_error * 100
            else:
                forward_variation_error[k] = np.nan
        else:
            forward_variation[k] = np.nan
            forward_variation_error[k] = np.nan
            forward_sample_sizes[k] = 0
            
    for k in range(n_interfaces - 1):
        q_values = []
        weights = []
        q_errors_k = []
        
        for i in range(k+2, n_interfaces):
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] >= min_samples:
                q_values.append(q_probs[i, k])
                weights.append(q_weights[i, k])
                if q_errors is not None and not np.isnan(q_errors[i, k]):
                    q_errors_k.append(q_errors[i, k])
                else:
                    binomial_error = np.sqrt(q_probs[i, k] * (1 - q_probs[i, k]) / q_weights[i, k]) if q_weights[i, k] > 0 else 0
                    q_errors_k.append(binomial_error)
        
        if len(q_values) >= 2:
            q_values = np.array(q_values)
            weights = np.array(weights)
            q_errors_arr = np.array(q_errors_k)
            total_samples = np.sum(weights)
            backward_sample_sizes[k] = total_samples
            std_dev = np.std(q_values)
            backward_variation[k] = std_dev * 100
            
            n = len(q_values)
            mean_q = np.mean(q_values)
            if std_dev > 0 and not np.any(np.isnan(q_errors_arr)):
                partial_derivs = (q_values - mean_q) / ((n - 1) * std_dev)
                std_error_squared = np.sum((partial_derivs * q_errors_arr) ** 2)
                std_error = np.sqrt(std_error_squared)
                backward_variation_error[k] = std_error * 100
            else:
                backward_variation_error[k] = np.nan
        else:
            backward_variation[k] = np.nan
            backward_variation_error[k] = np.nan
            backward_sample_sizes[k] = 0
            
    return {
        'forward_variation': forward_variation,
        'forward_variation_error': forward_variation_error, 
        'backward_variation': backward_variation,
        'backward_variation_error': backward_variation_error, 
        'forward_sample_sizes': forward_sample_sizes,
        'backward_sample_sizes': backward_sample_sizes
    }

def estimate_free_energy_differences(interfaces, q_matrix, q_weights=None, min_samples=5, account_for_distances=True):
    n_interfaces = len(interfaces)
    delta_G = np.zeros((n_interfaces, n_interfaces))
    delta_G.fill(np.nan)
    has_physical_distances = True # Assume useful if len > 1
    if not isinstance(interfaces[0], (int, float)):
        has_physical_distances = False

    def is_valid_q(i, k):
        return (not np.isnan(q_matrix[i, k]) and
                (q_weights is None or q_weights[i, k] >= min_samples) and
                abs(i-k) >= 2)
    
    for i in range(n_interfaces - 1):
        forward_estimate = None
        backward_estimate = None
        
        for start in range(i-1, -1, -1):
            if is_valid_q(start, i+1):
                q_fw = q_matrix[start, i+1]
                if has_physical_distances and i > 0:
                    dist_i_to_ip1 = interfaces[i+1] - interfaces[i]
                    dist_im1_to_i = interfaces[i] - interfaces[i-1]
                    geo_q = dist_im1_to_i / (dist_i_to_ip1 + dist_im1_to_i) if (dist_i_to_ip1 + dist_im1_to_i) > 0 else 1.0
                    if 0 < q_fw < 1 and 0 < geo_q < 1:
                        dG_obs = -np.log(q_fw / (1 - q_fw))
                        dG_geo = -np.log(geo_q / (1 - geo_q))
                        dG = dG_obs - dG_geo
                    else:
                         dG = 0 # Safe fallback
                else:
                    if 0 < q_fw < 1:
                        dG = -np.log(q_fw / (1 - q_fw))
                    else:
                        dG = 0
                forward_estimate = np.nan_to_num(dG, posinf=6.0, neginf=-6.0)
                break
        
        for start in range(i+2, n_interfaces):
            if is_valid_q(start, i):
                q_bw = q_matrix[start, i]
                if has_physical_distances and i+1 < n_interfaces-1:
                    dist_ip1_to_i = interfaces[i+1] - interfaces[i]
                    dist_ip2_to_ip1 = interfaces[i+2] - interfaces[i+1]
                    geo_q = dist_ip2_to_ip1 / (dist_ip2_to_ip1 + dist_ip1_to_i) if (dist_ip2_to_ip1 + dist_ip1_to_i) > 0 else 1.0
                    if 0 < q_bw < 1 and 0 < geo_q < 1:
                        dG_obs = np.log(q_bw / (1 - q_bw))
                        dG_geo = np.log(geo_q / (1 - geo_q))
                        dG = dG_obs - dG_geo
                    else:
                        dG = 0
                else:
                    if 0 < q_bw < 1:
                        dG = np.log(q_bw / (1 - q_bw))
                    else:
                        dG = 0
                backward_estimate = np.nan_to_num(dG, posinf=6.0, neginf=-6.0)
                break
        
        if forward_estimate is not None and backward_estimate is not None:
            combined_dG = (forward_estimate + backward_estimate) / 2.0
            delta_G[i, i+1] = combined_dG
            delta_G[i+1, i] = -combined_dG
        elif forward_estimate is not None:
            delta_G[i, i+1] = forward_estimate
            delta_G[i+1, i] = -forward_estimate
        elif backward_estimate is not None:
            delta_G[i, i+1] = backward_estimate
            delta_G[i+1, i] = -backward_estimate
            
    return delta_G

def calculate_diffusive_reference(interfaces, q_matrix, q_weights=None, min_samples=5, account_for_distances=True):
    n_interfaces = len(interfaces)
    diffusive_q = np.zeros((n_interfaces, n_interfaces))
    diffusive_q.fill(np.nan)
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples, account_for_distances)
    has_physical_distances = True
    if not isinstance(interfaces[0], (int, float)):
        has_physical_distances = False

    for i in range(n_interfaces):
        diffusive_q[i, i] = 0.0
    
    for k in range(1, n_interfaces):
        for i in range(k):
            if i == k-1:
                diffusive_q[i, k] = 1.0
            else:
                ref_prob = 0.5
                if not np.isnan(delta_G[k-1, k]):
                    if has_physical_distances and k > 1:
                        dist_km1_to_k = interfaces[k] - interfaces[k-1]
                        dist_km2_to_km1 = interfaces[k-1] - interfaces[k-2] if k > 1 else dist_km1_to_k
                        if (dist_km1_to_k + dist_km2_to_km1) > 0:
                            geo_q = dist_km2_to_km1 / (dist_km1_to_k + dist_km2_to_km1)
                        else:
                            geo_q = 0.5
                        geo_q = max(0.001, min(0.999, geo_q)) 
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]) * (1-geo_q)/geo_q)
                    else:
                        ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]))
                diffusive_q[i, k] = ref_prob
    
    for k in range(n_interfaces-1):
        for i in range(k+1, n_interfaces):
            if i == k+1:
                diffusive_q[i, k] = 1.0
            else:
                ref_prob = 0.5
                if not np.isnan(delta_G[k, k+1]):
                    if has_physical_distances and k+2 < n_interfaces:
                        dist_k_to_kp1 = interfaces[k+1] - interfaces[k]
                        dist_kp1_to_kp2 = interfaces[k+2] - interfaces[k+1] if k+2 < n_interfaces else dist_k_to_kp1
                        if (dist_k_to_kp1 + dist_kp1_to_kp2) > 0:
                            geo_q = dist_kp1_to_kp2 / (dist_k_to_kp1 + dist_kp1_to_kp2)
                        else:
                             geo_q = 0.5
                        geo_q = max(0.001, min(0.999, geo_q))
                        ref_prob = 1.0 / (1.0 + np.exp(-delta_G[k, k+1]) * (1-geo_q)/geo_q)
                    else:
                        ref_prob = 1.0 / (1.0 + np.exp(-delta_G[k, k+1]))
                diffusive_q[i, k] = ref_prob
                
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(diffusive_q[i, k]):
                diffusive_q[i, k] = max(0.0, min(1.0, diffusive_q[i, k]))
    return diffusive_q

def analyze_momentum_vs_free_energy(interfaces, q_matrix, q_weights=None, min_samples=5, momentum_threshold=0.2):
    n_interfaces = len(interfaces)
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples, account_for_distances=True)
    diffusive_q = calculate_diffusive_reference(interfaces, q_matrix, q_weights, min_samples, account_for_distances=True)
    momentum_effects = np.zeros_like(q_matrix)
    momentum_effects.fill(np.nan)
    
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]) and not np.isnan(diffusive_q[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    momentum_effects[i, k] = (q_matrix[i, k] - diffusive_q[i, k])
    
    momentum_significance = np.zeros_like(q_matrix, dtype=bool)
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(momentum_effects[i, k]):
                momentum_significance[i, k] = abs(momentum_effects[i, k]) > momentum_threshold
    
    pair_classification = []
    for i in range(n_interfaces-1):
        forward_effect = momentum_effects[i-1, i+1] if i > 0 and not np.isnan(momentum_effects[i-1, i+1]) else 0
        backward_effect = momentum_effects[i+2, i] if i+2 < n_interfaces and not np.isnan(momentum_effects[i+2, i]) else 0
        forward_significant = momentum_significance[i-1, i+1] if i > 0 else False
        backward_significant = momentum_significance[i+2, i] if i+2 < n_interfaces else False
        avg_effect = (abs(forward_effect) + abs(backward_effect)) / 2 if (i > 0 and i+2 < n_interfaces) else (abs(forward_effect) if i > 0 else abs(backward_effect))
        
        if forward_significant or backward_significant:
            if abs(forward_effect + backward_effect) < 0.2 * (abs(forward_effect) + abs(backward_effect)):
                pair_classification.append("symmetric_momentum")
            elif avg_effect > momentum_threshold * 2:
                pair_classification.append("strong_momentum")
            else:
                pair_classification.append("momentum_dominated")
        else:
            pair_classification.append("free_energy_dominated")
    
    avg_abs_momentum = np.nanmean(np.abs(momentum_effects))
    sum_momentum = np.nansum(momentum_effects)
    
    if np.isnan(avg_abs_momentum):
        overall_classification = "insufficient_data"
    elif avg_abs_momentum < momentum_threshold:
        overall_classification = "free_energy_dominated"
    elif abs(sum_momentum) < 0.2 * np.nansum(np.abs(momentum_effects)):
        overall_classification = "symmetric_momentum_dominated"
    else:
        overall_classification = "directional_momentum_dominated"
        
    avg_abs_free_energy = np.nanmean(np.abs(delta_G))
    avg_probabilities = np.nanmean([np.nanmean(q_matrix[i, :]) for i in range(n_interfaces)])
    
    return {
        'free_energy_differences': delta_G,
        'diffusive_probabilities': diffusive_q,
        'momentum_effects': momentum_effects,
        'momentum_significance': momentum_significance,
        'classification': pair_classification,
        'overall_classification': overall_classification,
        'avg_momentum_effect': avg_abs_momentum,
        'avg_free_energy': avg_abs_free_energy,
        'avg_probabilities': avg_probabilities
    }

def analyze_turn_persistence(q_matrix, q_weights=None, min_samples=5):
    n_interfaces = q_matrix.shape[0]
    avg_turn_skip = np.zeros(n_interfaces)
    avg_turn_skip.fill(np.nan)
    for i in range(n_interfaces):
        valid_distances = []
        valid_weights = []
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    distance = abs(k - i)
                    weight = q_matrix[i, k]
                    sample_weight = q_weights[i, k] if q_weights is not None else 1
                    valid_distances.append(distance)
                    valid_weights.append(weight * sample_weight)
        if valid_distances:
            avg_turn_skip[i] = np.average(valid_distances, weights=valid_weights)
            
    turn_asymmetry = np.zeros(n_interfaces)
    turn_asymmetry.fill(np.nan)
    for i in range(1, n_interfaces-1):
        forward_prob = 0
        backward_prob = 0
        forward_count = 0
        backward_count = 0
        for k in range(n_interfaces):
            if k > i and not np.isnan(q_matrix[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    forward_prob += q_matrix[i, k]
                    forward_count += 1
            elif k < i and not np.isnan(q_matrix[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    backward_prob += q_matrix[i, k]
                    backward_count += 1
        if forward_count > 0: forward_prob /= forward_count
        if backward_count > 0: backward_prob /= backward_count
        if forward_prob + backward_prob > 0:
            turn_asymmetry[i] = (forward_prob - backward_prob) / (forward_prob + backward_prob)
            
    transition_sharpness = np.zeros(n_interfaces)
    transition_sharpness.fill(np.nan)
    for i in range(n_interfaces):
        valid_probs = []
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    valid_probs.append(q_matrix[i, k])
        if len(valid_probs) > 1:
            norm_probs = np.array(valid_probs) / np.sum(valid_probs)
            entropy = -np.sum(norm_probs * np.log(norm_probs + 1e-10))
            max_entropy = np.log(len(valid_probs))
            if max_entropy > 0:
                transition_sharpness[i] = 1.0 - entropy / max_entropy
            else:
                transition_sharpness[i] = np.nan
    return {
        'avg_turn_skip': avg_turn_skip,
        'turn_asymmetry': turn_asymmetry,
        'transition_sharpness': transition_sharpness
    }

def analyze_memory_vs_free_energy_effects(interfaces, q_matrix, q_weights=None, min_samples=5, memory_threshold=0.3):
    n_interfaces = len(interfaces)
    delta_G = estimate_free_energy_differences(interfaces, q_matrix, q_weights, min_samples)
    predicted_q = np.zeros_like(q_matrix)
    predicted_q.fill(np.nan)
    for i in range(n_interfaces): predicted_q[i, i] = 0.0
    
    for k in range(1, n_interfaces):
        ref_prob = 0.5
        if not np.isnan(delta_G[k-1, k]):
            ref_prob = 1.0 / (1.0 + np.exp(delta_G[k-1, k]))
        for i in range(k):
            if i == k-1: predicted_q[i, k] = 1.0
            else: predicted_q[i, k] = ref_prob
            
    for k in range(n_interfaces-1):
        ref_prob = 0.5
        if not np.isnan(delta_G[k+1, k]):
            ref_prob = 1.0 / (1.0 + np.exp(delta_G[k+1, k]))
        for i in range(k+1, n_interfaces):
            if i == k+1: predicted_q[i, k] = 1.0
            else: predicted_q[i, k] = ref_prob
            
    memory_effects = np.zeros_like(q_matrix)
    memory_effects.fill(np.nan)
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i != k and not np.isnan(q_matrix[i, k]) and not np.isnan(predicted_q[i, k]):
                if q_weights is None or q_weights[i, k] >= min_samples:
                    if predicted_q[i, k] > 0:
                        memory_effects[i, k] = (q_matrix[i, k] - predicted_q[i, k]) / predicted_q[i, k]
    
    memory_significance = np.zeros_like(q_matrix, dtype=bool)
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if not np.isnan(memory_effects[i, k]):
                memory_significance[i, k] = abs(memory_effects[i, k]) > memory_threshold
                
    turn_persistence = analyze_turn_persistence(q_matrix, q_weights, min_samples)
    interface_classification = {}
    for i in range(n_interfaces):
        memory_count = np.sum(memory_significance[i, :])
        total_transitions = np.sum(~np.isnan(memory_effects[i, :]))
        if total_transitions == 0: interface_classification[i] = "insufficient_data"
        elif memory_count / total_transitions > 0.5: interface_classification[i] = "memory_dominated"
        elif memory_count / total_transitions < 0.2: interface_classification[i] = "free_energy_dominated"
        else: interface_classification[i] = "mixed"
        
    total_significant_memory = np.sum(memory_significance)
    total_valid_transitions = np.sum(~np.isnan(memory_effects))
    if total_valid_transitions == 0: overall_classification = "insufficient_data"
    elif total_significant_memory / total_valid_transitions > 0.5: overall_classification = "memory_dominated"
    elif total_significant_memory / total_valid_transitions < 0.2: overall_classification = "free_energy_dominated"
    else: overall_classification = "mixed"
    
    forward_memory = 0; forward_count = 0
    backward_memory = 0; backward_count = 0
    for i in range(n_interfaces):
        for k in range(n_interfaces):
            if i < k and not np.isnan(memory_effects[i, k]):
                forward_memory += abs(memory_effects[i, k]); forward_count += 1
            elif i > k and not np.isnan(memory_effects[i, k]):
                backward_memory += abs(memory_effects[i, k]); backward_count += 1
    avg_forward_memory = forward_memory / forward_count if forward_count > 0 else 0
    avg_backward_memory = backward_memory / backward_count if backward_count > 0 else 0
    
    return {
        'free_energy_differences': delta_G,
        'predicted_q_matrix': predicted_q,
        'memory_effects': memory_effects,
        'memory_significance': memory_significance,
        'interface_classification': interface_classification,
        'overall_classification': overall_classification,
        'avg_forward_memory': avg_forward_memory,
        'avg_backward_memory': avg_backward_memory,
        'forward_vs_backward': 'forward_dominated' if avg_forward_memory > avg_backward_memory * 1.5 else ('backward_dominated' if avg_backward_memory > avg_forward_memory * 1.5 else 'balanced'),
        'turn_based_metrics': turn_persistence
    }

def visualize_turn_based_analysis(analysis_results, interfaces, q_matrix, q_weights=None):
    n_interfaces = len(interfaces)
    delta_G = analysis_results['free_energy_differences']
    predicted_q = analysis_results['predicted_q_matrix']
    memory_effects = analysis_results['memory_effects']
    memory_significance = analysis_results['memory_significance']
    
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    overall = analysis_results['overall_classification'].replace("_", " ").title()
    forward_vs_backward = analysis_results['forward_vs_backward'].replace("_", " ").title()
    title_text = f'Turn-Based Memory vs Free Energy Analysis\\nOverall: {overall} ({forward_vs_backward})'
    fig.text(0.5, 0.98, title_text, ha='center', va='top', fontsize=16, weight='bold')
    
    ax1 = fig.add_subplot(gs[0, 0])
    cumulative_G = np.zeros(n_interfaces)
    for i in range(1, n_interfaces):
        valid_path = True
        for j in range(i):
            if np.isnan(delta_G[j, j+1]):
                valid_path = False; break
            cumulative_G[i] += delta_G[j, j+1]
        if not valid_path: cumulative_G[i] = np.nan
        
    ax1.plot(interfaces, cumulative_G, 'o-', linewidth=2, color='blue')
    for i, (pos, g) in enumerate(zip(interfaces, cumulative_G)):
        if not np.isnan(g):
            ax1.plot(pos, g, 'o', markersize=8, color='blue')
            ax1.text(pos, g + 0.1, f"{g:.2f}", ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel('Interface Position')
    ax1.set_ylabel('Free Energy G (kT)')
    ax1.set_title('Free Energy Profile from Turn-Based Transitions', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    cmap_memory = LinearSegmentedColormap.from_list('memory_effect', [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    masked_memory = np.ma.masked_invalid(memory_effects)
    max_effect = np.nanmax(np.abs(memory_effects)) if not np.all(np.isnan(memory_effects)) else 1.0
    im = ax2.imshow(masked_memory, cmap=cmap_memory, vmin=-max_effect, vmax=max_effect, interpolation='none', aspect='auto')
    cbar = fig.colorbar(im, ax=ax2, label='Memory Effect (q - q_predicted) / q_predicted')
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_effects[i, j]):
                sig_mark = '*' if memory_significance[i, j] else ''
                text = f"{memory_effects[i, j]:.2f}{sig_mark}"
                color = 'black' if abs(memory_effects[i, j]) < 0.5 * max_effect else 'white'
                ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    ax2.set_title('Memory Effects: Deviation from Free Energy Model', fontsize=12)
    
    ax3 = fig.add_subplot(gs[1, 0])
    valid_points = []
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j and not np.isnan(q_matrix[i, j]) and not np.isnan(predicted_q[i, j]):
                valid_points.append((predicted_q[i, j], q_matrix[i, j], memory_significance[i, j], f"{i}->{j}"))
    if valid_points:
        x_vals, y_vals, significance, labels = zip(*valid_points)
        ax3.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7)
        for x, y, sig, label in zip(x_vals, y_vals, significance, labels):
            color = 'red' if sig else 'blue'
            ax3.scatter(x, y, color=color, s=50, alpha=0.7)
    ax3.set_xlabel('Predicted Turn Probability (Free Energy Model)')
    ax3.set_ylabel('Observed Turn Probability')
    ax3.set_title('Observed vs Predicted Turn Probabilities', fontsize=12)
    ax3.grid(True, alpha=0.3)
    return fig

def ploc_repptis_from_staples(pes, interfaces, n_int=None, staple_weights=None):
    """Fallback stub for ploc_repptis_from_staples.

    If `pes` is None or the function cannot compute REPPTIS values, return
    dictionaries with NaN placeholders so plotting still works and matches the
    original layout. If a proper `pes` is provided and the real implementation
    is importable, attempt to delegate to it.
    """
    n = len(interfaces)
    # Default placeholders (keys mimic original structure)
    plocs_repptis = {k: {"LMR": np.full(n, np.nan), "RMR": np.full(n, np.nan)} for k in range(n)}
    plocs_istar = {k: {"LMR": np.full(n, np.nan), "RMR": np.full(n, np.nan)} for k in range(n)}

    # Try to delegate to tistools implementation if available
    try:
        from tistools.lib.istar_analysis import ploc_repptis_from_staples as _real_ploc
        return _real_ploc(pes, interfaces, n_int=n_int, staple_weights=staple_weights)
    except Exception:
        return plocs_repptis, plocs_istar


def plot_memory_analysis(q_tot, p, interfaces=None, q_errors=None, pes=None):
    """
    Generate comprehensive visualizations for memory effect analysis (weight-matrix-first API).

    Signature supports either:
      - plot_memory_analysis(q_tot, p, interfaces=..., q_errors=..., pes=None)
      - (backward-compatible) plot_memory_analysis(pes, q_tot, p, ...)

    Parameters:
    -----------
    q_tot : numpy.ndarray
        A matrix with shape [2, n_interfaces, n_interfaces] where:
        - q_tot[0][i][k]: conditional crossing probabilities
        - q_tot[1][i][k]: sample counts for each calculation
    p : numpy.ndarray
        Transition probability matrix between interfaces
    interfaces : list, optional
        The interface positions for axis labeling. If None, uses sequential indices.
    q_errors : ndarray, optional
        Error estimates for transition probabilities
    pes : list, optional
        PathEnsemble objects (optional; plotting will work if omitted)

    Returns:
    -------
    tuple
        (fig1, fig2, fig3) matplotlib.figure.Figure objects
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    import seaborn as sns
    # Add legend
    from matplotlib.lines import Line2D
    
    # Extract the probability matrix and weights matrix from q_tot
    q_probs = q_tot[0]
    q_weights = q_tot[1]
    n_interfaces = q_probs.shape[0]
    
    if interfaces is None:
        interfaces = list(range(n_interfaces))
        is_equidistant = True
    else:
        # Check if interfaces are equidistant
        if len(interfaces) > 2:
            diffs = np.diff(interfaces)
            is_equidistant = np.allclose(diffs, diffs[0], rtol=0.05)
        else:
            is_equidistant = True
    
    # Generate more descriptive state labels
    state_labels = generate_state_labels(n_interfaces)

    try:
        from tistools.lib.istar_analysis import construct_M_istar as _construct_M_istar
        M = _construct_M_istar(p, 2*n_interfaces, n_interfaces)
    except Exception:
        M = None  # construct_M_istar not available in this module's scope; proceed without it
    
    # Calculate diffusive reference probabilities based on interface spacing
    diff_ref = calculate_diffusive_reference(interfaces, q_tot[0], q_tot[1])
    plocs_repptis, plocs_istar = ploc_repptis_from_staples(pes, interfaces, n_int=n_interfaces)
    
    # Function to generate high-contrast colors for plots
    def generate_high_contrast_colors(n):
        if n <= 1:
            return ["#1f77b4"]  # Default blue for single item
        
        if n <= 10:
            # Viridis with enhanced spacing for better contrast
            viridis_cmap = plt.cm.get_cmap('viridis')
            return [colors.to_hex(viridis_cmap(i/(n-1) if n > 1 else 0.5)) for i in range(n)]
        else:
            # For more interfaces, use viridis with adjusted spacing
            cmap1 = plt.cm.get_cmap('viridis')
            
            # Get colors with deliberate spacing for better contrast
            colors_list = []
            for i in range(n):
                # Distribute colors with slight variations in spacing
                # This avoids adjacent indices having too similar colors
                pos = (i / max(1, n-1)) * 0.85 + 0.1  # Scale to range 0.1-0.95
                
                # Introduce small oscillations in color position for adjacent indices
                if i % 2 == 1:
                    pos = min(0.95, pos + 0.05)
                    
                colors_list.append(colors.to_hex(cmap1(pos)))
                
            return colors_list

    # ================ Figure 1: Matrix Heatmaps ================
    fig1 = plt.figure(figsize=(18, 7))
    gs1 = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    
    # Create custom colormap for memory effect heatmap
    cmap_memory = LinearSegmentedColormap.from_list('memory_effect', 
                                                  [(0, 'blue'), (0.5, 'white'), (1, 'red')], N=256)
    
    # Plot 1.1: Memory Effect Matrix (q_probs)
    ax1 = fig1.add_subplot(gs1[0])
    
    # Calculate memory effect as deviation from diffusive reference
    memory_effect = np.zeros_like(q_probs)
    memory_effect.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(q_probs[i, j]) and not np.isnan(diff_ref[i, j]):
                memory_effect[i, j] = q_probs[i, j] - diff_ref[i, j]
    
    # Create diverging colormap centered at 0
    max_effect = np.nanmax(np.abs(memory_effect))
    
    masked_data = np.ma.masked_invalid(memory_effect)  # Mask NaN values
    im1 = ax1.imshow(masked_data, cmap=cmap_memory, vmin=-max_effect, vmax=max_effect, 
                    interpolation='none', aspect='auto')
    
    # Add colorbar
    cbar1 = fig1.colorbar(im1, ax=ax1, label='Memory Effect (q - q_diff)')
    
    # Add reference line at 0
    cbar1.ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    cbar1.ax.text(1.5, 0.0, '0 (diffusive)', va='center', ha='left', fontsize=9)
    
    # Add annotations with more compact formatting
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_effect[i, j]) and not np.ma.is_masked(masked_data[i, j]):
                weight = q_weights[i, j]
                # More compact format: actual/diff
                text = f"{q_probs[i, j]:.2f}/{diff_ref[i, j]:.2f}" if weight > 0 else "N/A"
                # Only show count if it's significant
                if weight > 10:
                    text += f"\n{int(weight)}"
                color = 'black' if abs(memory_effect[i, j]) < 0.3 else 'white'
                ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=7)
    
    # Set ticks and labels using state labels
    ax1.set_xticks(np.arange(n_interfaces))
    ax1.set_yticks(np.arange(n_interfaces))
    ax1.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    ax1.set_xlabel('Target Turn at k')
    ax1.set_ylabel('Starting Turn at i')
    ax1.set_title('Memory Effect Matrix: q(i,k) - q_diffuse(i,k)', fontsize=12)
    
    # Plot 1.2: Memory Effect Ratio
    ax2 = fig1.add_subplot(gs1[1])
    
    # Calculate memory effect ratio: ratio of actual prob to diffusive prob
    memory_ratio = np.zeros_like(q_probs)
    memory_ratio.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if (not np.isnan(q_probs[i, j]) and not np.isnan(diff_ref[i, j]) and
                diff_ref[i, j] > 0 and diff_ref[i, j] < 1):
                memory_ratio[i, j] = q_probs[i, j] / diff_ref[i, j]
    
    # Plot heatmap with logarithmic scale
    im2 = ax2.imshow(memory_ratio, cmap='RdBu_r', norm=colors.LogNorm(vmin=0.1, vmax=10))
    
    # Add colorbar
    cbar2 = fig1.colorbar(im2, ax=ax2, label='Probability Ratio q/q_diffuse [log scale]')
    
    # Add annotations for ratio values - more compact
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_ratio[i, j]) and q_weights[i, j] > 5:
                text_color = 'black'
                if memory_ratio[i, j] > 5 or memory_ratio[i, j] < 0.2:
                    text_color = 'white'
                ax2.text(j, i, f"{memory_ratio[i, j]:.1f}", ha='center', va='center', 
                       color=text_color, fontsize=7)
    
    ax2.set_xlabel('Target Turn at  k')
    ax2.set_ylabel('Starting Turn at i')
    ax2.set_title('Memory Effect Ratio: Deviation from Diffusive Behavior', fontsize=12)
    ax2.set_xticks(range(n_interfaces))
    ax2.set_yticks(range(n_interfaces))
    ax2.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax2.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Plot 1.3: Memory Asymmetry
    ax3 = fig1.add_subplot(gs1[2])
    
    # Calculate memory asymmetry for pairs of interfaces (i, j) using the p matrix
    memory_asymmetry = np.zeros_like(p)
    memory_asymmetry.fill(np.nan)
    
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if i != j:
                # Asymmetry is the difference between forward and backward transition probabilities
                memory_asymmetry[i, j] = p[i, j] - p[j, i]
    
    # Plot heatmap
    im3 = ax3.imshow(memory_asymmetry, cmap='RdBu', vmin=-0.5, vmax=0.5)
    
    # Add colorbar
    cbar3 = fig1.colorbar(im3, ax=ax3, label='Probability Asymmetry (i→j vs j→i)')
    
    # Add annotations - more compact
    for i in range(n_interfaces):
        for j in range(n_interfaces):
            if not np.isnan(memory_asymmetry[i, j]):
                text_color = 'black'
                if abs(memory_asymmetry[i, j]) > 0.3:
                    text_color = 'white'
                ax3.text(j, i, f"{memory_asymmetry[i, j]:.2f}", ha='center', va='center', 
                       color=text_color, fontsize=7)
    
    ax3.set_xlabel('Target Turn at j')
    ax3.set_ylabel('Starting Turn at i')
    ax3.set_title('Memory Asymmetry: Forward vs. Backward Transitions', fontsize=12)
    ax3.set_xticks(range(n_interfaces))
    ax3.set_yticks(range(n_interfaces))
    ax3.set_xticklabels([f"{i}" for i in range(n_interfaces)])
    ax3.set_yticklabels([f"{i}" for i in range(n_interfaces)])
    
    # Add explanatory text that includes info about non-equidistant interfaces
    if is_equidistant:
        desc_text = """
        Memory Effect Matrix: Shows deviations from diffusive behavior.
        In a purely diffusive process, all values would be 0.
        Values > 0 (red) indicate bias toward crossing, < 0 (blue) indicate bias toward returning.
        """
    else:
        desc_text = """
        Memory Effect Matrix: Shows deviations from diffusive behavior.
        Due to non-equidistant interfaces, the diffusive reference varies for each transition.
        Values > 0 (red) indicate bias toward crossing, < 0 (blue) indicate bias toward returning.
        """
    fig1.text(0.02, 0.02, desc_text, fontsize=10, wrap=True)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig1.suptitle('TIS Memory Effect Analysis - Matrix Representations' + 
                (' (Non-equidistant Interfaces)' if not is_equidistant else ''), fontsize=14)
    # (duplicate Figure 2 stub removed — consolidated below)

    
    # ============================================================================
    # FIGURE 2: Forward and Backward Transition Probabilities with Memory Retention
    # ============================================================================
    fig2 = plt.figure(figsize=(18, 12))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.25)
    
    # Forward transitions (L→R)
    ax4 = fig2.add_subplot(gs2[0, 0])
    colors_fwd = plt.cm.tab10(np.linspace(0, 1, n_interfaces))
    for k in range(1, n_interfaces):
        start_is = []
        vals = []
        refs = []
        for i in range(k):
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                start_is.append(interfaces[i])
                vals.append(q_probs[i, k])
                refs.append(diff_ref[i, k])
        if vals:
            ax4.plot(start_is, vals, 'o-', color=colors_fwd[k], label=f'To $\lambda_{k}$', 
                    linewidth=2, markersize=8, alpha=0.8)
            ax4.plot(start_is, refs, '--', color=colors_fwd[k], alpha=0.4, linewidth=1.5)
    
    ax4.set_xlabel('Start Interface $\lambda_i$', fontsize=12)
    ax4.set_ylabel('Transition Probability q(i→k)', fontsize=12)
    ax4.set_title('Forward Transition Probabilities (L→R)', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    sns.despine(ax=ax4)
    
    # Backward transitions (R→L)
    ax5 = fig2.add_subplot(gs2[0, 1])
    colors_bwd = plt.cm.tab10(np.linspace(0, 1, n_interfaces))
    for k in range(n_interfaces - 1):
        start_is = []
        vals = []
        refs = []
        for i in range(k + 1, n_interfaces):
            if not np.isnan(q_probs[i, k]) and q_weights[i, k] > 5:
                start_is.append(interfaces[i])
                vals.append(q_probs[i, k])
                refs.append(diff_ref[i, k])
        if vals:
            ax5.plot(start_is, vals, 'o-', color=colors_bwd[k], label=f'To $\lambda_{k}$',
                    linewidth=2, markersize=8, alpha=0.8)
            ax5.plot(start_is, refs, '--', color=colors_bwd[k], alpha=0.4, linewidth=1.5)
    
    ax5.set_xlabel('Start Interface $\lambda_i$', fontsize=12)
    ax5.set_ylabel('Transition Probability q(i→k)', fontsize=12)
    ax5.set_title('Backward Transition Probabilities (R→L)', fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.05)
    sns.despine(ax=ax5)
    
    # Memory retention bar charts
    memory_index = calculate_memory_effect_index(q_probs, q_weights, q_errors)
    
    # Forward memory retention
    ax6 = fig2.add_subplot(gs2[1, 0])
    fwd_var = memory_index['forward_variation']
    fwd_err = memory_index['forward_variation_error']
    fwd_samples = memory_index['forward_sample_sizes']
    
    valid_fwd = ~np.isnan(fwd_var)
    x_fwd = np.arange(n_interfaces)[valid_fwd]
    bars1 = ax6.bar(x_fwd, fwd_var[valid_fwd], color='steelblue', alpha=0.7, edgecolor='black')
    if not np.all(np.isnan(fwd_err)):
        ax6.errorbar(x_fwd, fwd_var[valid_fwd], yerr=fwd_err[valid_fwd], fmt='none', 
                    ecolor='black', capsize=3, alpha=0.5)
    
    for i, (idx, samples) in enumerate(zip(x_fwd, fwd_samples[valid_fwd])):
        if samples > 0:
            ax6.text(idx, fwd_var[idx] + (fwd_err[idx] if not np.isnan(fwd_err[idx]) else 0) + 0.5,
                    f'n={samples}', ha='center', va='bottom', fontsize=8)
    
    ax6.set_xlabel('Target Interface k', fontsize=12)
    ax6.set_ylabel('Memory Retention (% std dev)', fontsize=12)
    ax6.set_title('Forward Memory Retention', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xticks(range(n_interfaces))
    sns.despine(ax=ax6)
    
    # Backward memory retention
    ax7 = fig2.add_subplot(gs2[1, 1])
    bwd_var = memory_index['backward_variation']
    bwd_err = memory_index['backward_variation_error']
    bwd_samples = memory_index['backward_sample_sizes']
    
    valid_bwd = ~np.isnan(bwd_var)
    x_bwd = np.arange(n_interfaces)[valid_bwd]
    bars2 = ax7.bar(x_bwd, bwd_var[valid_bwd], color='coral', alpha=0.7, edgecolor='black')
    if not np.all(np.isnan(bwd_err)):
        ax7.errorbar(x_bwd, bwd_var[valid_bwd], yerr=bwd_err[valid_bwd], fmt='none',
                    ecolor='black', capsize=3, alpha=0.5)
    
    for i, (idx, samples) in enumerate(zip(x_bwd, bwd_samples[valid_bwd])):
        if samples > 0:
            ax7.text(idx, bwd_var[idx] + (bwd_err[idx] if not np.isnan(bwd_err[idx]) else 0) + 0.5,
                    f'n={samples}', ha='center', va='bottom', fontsize=8)
    
    ax7.set_xlabel('Target Interface k', fontsize=12)
    ax7.set_ylabel('Memory Retention (% std dev)', fontsize=12)
    ax7.set_title('Backward Memory Retention', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_xticks(range(n_interfaces))
    sns.despine(ax=ax7)
    
    fig2.suptitle('Transition Probabilities and Memory Retention Analysis', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    # ============================================================================
    # FIGURE 3: Free Energy and Momentum Effects Analysis
    # ============================================================================
    fig3 = plt.figure(figsize=(18, 12))
    gs3 = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
    
    # Free energy profile (spanning top row)
    ax8 = fig3.add_subplot(gs3[0, :])
    delta_G = estimate_free_energy_differences(interfaces, q_probs, q_weights)
    
    # Reconstruct cumulative free energy profile
    G_profile = np.zeros(n_interfaces)
    for i in range(n_interfaces - 1):
        if not np.isnan(delta_G[i, i+1]):
            G_profile[i+1] = G_profile[i] + delta_G[i, i+1]
        else:
            G_profile[i+1] = G_profile[i]
    
    ax8.plot(interfaces, G_profile, 'o-', color='darkgreen', linewidth=2.5, 
             markersize=10, label='Free Energy Profile')
    ax8.fill_between(interfaces, G_profile, alpha=0.2, color='green')
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax8.set_xlabel('Interface Position $\lambda$', fontsize=12)
    ax8.set_ylabel('Free Energy $G(\lambda)$ [kT]', fontsize=12)
    ax8.set_title('Free Energy Profile from Transition Probabilities', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=11)
    sns.despine(ax=ax8)
    
    # Observed vs Diffusive comparison
    ax9 = fig3.add_subplot(gs3[1, 0])
    momentum_results = analyze_momentum_vs_free_energy(interfaces, q_probs, q_weights)
    
    # Plot for representative transitions (show all available start→target pairs)
    for k in range(1, n_interfaces):
        obs_vals = []
        diff_vals = []
        start_positions = []
        for i in range(0, k):
            if not np.isnan(q_probs[i, k]) and not np.isnan(diff_ref[i, k]) and q_weights[i, k] > 5:
                obs_vals.append(q_probs[i, k])
                diff_vals.append(diff_ref[i, k])
                start_positions.append(i)
        
        if obs_vals:
            ax9.scatter(diff_vals, obs_vals, s=100, alpha=0.7, label=f'To $\lambda_{k}$')
    
    # Add diagonal reference line
    max_val = max(ax9.get_xlim()[1], ax9.get_ylim()[1])
    ax9.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    ax9.set_xlabel('Diffusive Probability q_diff', fontsize=12)
    ax9.set_ylabel('Observed Probability q_obs', fontsize=12)
    ax9.set_title('Observed vs Diffusive Probabilities', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(0, 1.05)
    ax9.set_ylim(0, 1.05)
    sns.despine(ax=ax9)
    
    # Interface pair classification
    ax10 = fig3.add_subplot(gs3[1, 1])
    classifications = momentum_results['classification']
    
    # Create color map for classifications
    class_colors = {
        'free_energy_dominated': 'blue',
        'momentum_dominated': 'red',
        'symmetric_momentum': 'purple',
        'strong_momentum': 'darkred'
    }
    
    colors_list = [class_colors.get(c, 'gray') for c in classifications]
    x_pos = np.arange(len(classifications))
    
    bars = ax10.barh(x_pos, [1]*len(classifications), color=colors_list, alpha=0.7, edgecolor='black')
    ax10.set_yticks(x_pos)
    ax10.set_yticklabels([f'$\lambda_{i}$→$\lambda_{i+1}$' for i in range(len(classifications))], fontsize=10)
    ax10.set_xlabel('Classification', fontsize=12)
    ax10.set_title('Interface Pair Classification', fontsize=12, fontweight='bold')
    ax10.set_xlim(0, 1.2)
    ax10.set_xticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=label.replace('_', ' ').title())
                      for label, color in class_colors.items()]
    ax10.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    # Add text annotations
    for i, (pos, classif) in enumerate(zip(x_pos, classifications)):
        ax10.text(0.5, pos, classif.replace('_', ' ').title(), 
                 ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    sns.despine(ax=ax10, left=True)
    
    fig3.suptitle('Free Energy and Momentum Effects Analysis', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    return fig1, fig2, fig3
