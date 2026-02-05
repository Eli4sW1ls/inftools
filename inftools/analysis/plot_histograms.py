"""
Plot histograms and free energies from infretis simulation data.

This module provides functions to:
1. Plot WHAM-weighted histograms and free energies (0- and i+ ensembles separately)
2. Plot per-ensemble histograms and free energies
3. Compare results between multiple simulations
4. Compare results across different gamma values

Usage:
    inft plot-histograms -datadir histograms/
    inft compare-histograms sim1/histograms sim2/histograms -labels "Sim 1" "Sim 2"
    inft compare-gamma /path/to/system -save comparison.png
"""
import glob
import os
import re
from pathlib import Path
from typing import Annotated, Optional, List, Dict, Tuple
import numpy as np
import typer
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.cm as cm

# Try to import scienceplots for publication-ready figures
try:
    import scienceplots  # type: ignore
    SCIENCEPLOTS_AVAILABLE = True
except Exception:
    SCIENCEPLOTS_AVAILABLE = False

# Default color cycle
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def extract_system_info_from_path(path: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Extract system name, gamma, and subcycles from directory path.
    
    Looks for 'gamma' in path parts and takes the preceding folder as system name.
    Also tries to read subcycles from infretis.toml in the path.
    
    Returns
    -------
    Tuple[str, str, int]
        (system_name, gamma, subcycles) - any can be None if not detected
    """
    path = Path(path).resolve()
    parts = path.parts
    
    # Try to find gamma in path parts
    gamma = None
    gamma_idx = None
    for i, part in enumerate(parts):
        if part.lower().startswith("gamma"):
            gamma = part
            gamma_idx = i
            break
    
    # System name is the folder before gamma
    system_name = None
    if gamma_idx is not None and gamma_idx > 0:
        system_name = parts[gamma_idx - 1]
    
    # Try to find subcycles from infretis.toml
    subcycles = None
    # Walk up from datadir to find infretis.toml
    search_path = path
    for _ in range(5):  # Look up to 5 levels
        toml_path = search_path / "infretis.toml"
        if toml_path.exists():
            try:
                from inftools.misc.infinit_helper import read_toml
                toml_dict = read_toml(str(toml_path))
                if toml_dict:
                    subcycles = toml_dict.get("engine", {}).get("subcycles")
            except Exception:
                pass
            break
        search_path = search_path.parent
    
    return system_name, gamma, subcycles


def build_plot_filename(base_name: str, system_name: Optional[str] = None, 
                        gamma: Optional[str] = None, subcycles: Optional[int] = None) -> str:
    """
    Build a descriptive filename for plots.
    
    Parameters
    ----------
    base_name : str
        Base name like 'wham_histogram' or 'per_ensemble_histogram'
    system_name, gamma, subcycles : optional
        System info to include in filename
    
    Returns
    -------
    str
        Filename like 'system_gamma_subcyclesN_basename.png'
    """
    parts = []
    if system_name:
        parts.append(system_name)
    if gamma:
        parts.append(gamma)
    if subcycles is not None:
        parts.append(f"subcycles{subcycles}")
    parts.append(base_name)
    return "_".join(parts) + ".png"


def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a CSV file with header.

    Returns
    -------
    Tuple containing:
        - x: First column (order parameter)
        - data: Remaining columns
        - headers: Column headers
    """
    with open(filepath, "r") as f:
        header_line = f.readline().strip()
        headers = header_line.split(",")

    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    x = data[:, 0]
    y = data[:, 1:] if data.shape[1] > 1 else data[:, 0:1]

    return x, y, headers[1:] if len(headers) > 1 else headers


def detect_normalization(headers: List[str]) -> str:
    """
    Detect normalization type from CSV headers.
    
    Returns
    -------
    str
        'time', 'density', or 'none'
    """
    header_str = ",".join(headers).lower()
    if "time" in header_str:  # matches avg_time_per_path or time_per_binwidth
        return "time"
    elif "probability_density" in header_str:
        return "density"
    else:
        return "none"


def load_histogram_data(datadir: str) -> Dict:
    """
    Load all histogram data from a directory.

    Parameters
    ----------
    datadir : str
        Directory containing CSV files from compute_histograms.

    Returns
    -------
    dict
        Dictionary with loaded data.
    """
    datadir = Path(datadir)
    result = {"datadir": str(datadir), "normalization": "none"}

    # Load WHAM [i+] ensembles data (new format)
    # CSV format: order_parameter, histogram, probability
    wham_plus_hist_file = datadir / "wham_plus_histogram.csv"
    wham_plus_fe_file = datadir / "wham_plus_free_energy.csv"

    if wham_plus_hist_file.exists():
        x, y, headers = load_csv(wham_plus_hist_file)
        result["wham_plus_x"] = x
        # Use histogram column (index 0) for the actual values (time, density, or counts)
        result["wham_plus_hist"] = y[:, 0] if y.ndim > 1 else y.flatten()
        # Also store probability column separately if available
        if y.shape[1] > 1:
            result["wham_plus_prob"] = y[:, 1]
        # Detect normalization from headers
        result["normalization"] = detect_normalization(headers)
    if wham_plus_fe_file.exists():
        x, y, _ = load_csv(wham_plus_fe_file)
        result["wham_plus_fe"] = y[:, 0] if y.ndim > 1 else y.flatten()

    # Load [0-] ensemble data (new format)
    # CSV format: order_parameter, histogram, probability
    ens_0min_hist_file = datadir / "ens_0min_histogram.csv"
    ens_0min_fe_file = datadir / "ens_0min_free_energy.csv"

    if ens_0min_hist_file.exists():
        x, y, headers = load_csv(ens_0min_hist_file)
        result["ens_0min_x"] = x
        # Use histogram column (index 0) for the actual values (time, density, or counts)
        result["ens_0min_hist"] = y[:, 0] if y.ndim > 1 else y.flatten()
        # Also store probability column separately if available
        if y.shape[1] > 1:
            result["ens_0min_prob"] = y[:, 1]
    if ens_0min_fe_file.exists():
        x, y, _ = load_csv(ens_0min_fe_file)
        result["ens_0min_fe"] = y[:, 0] if y.ndim > 1 else y.flatten()

    # Fallback: Load old format (wham_histogram.csv) if new files don't exist
    wham_hist_file = datadir / "wham_histogram.csv"
    wham_fe_file = datadir / "wham_free_energy.csv"

    if wham_hist_file.exists() and "wham_plus_x" not in result:
        x, y, _ = load_csv(wham_hist_file)
        result["wham_x"] = x
        result["wham_hist"] = y.flatten()
    if wham_fe_file.exists() and "wham_plus_fe" not in result:
        x, y, _ = load_csv(wham_fe_file)
        result["wham_fe"] = y.flatten()

    # Load all-ensembles data
    all_hist_file = datadir / "all_ensembles_histogram.csv"
    all_fe_file = datadir / "all_ensembles_free_energy.csv"

    if all_hist_file.exists():
        x, y, headers = load_csv(all_hist_file)
        result["ens_x"] = x
        result["ens_hist"] = y
        result["ens_labels"] = headers

    if all_fe_file.exists():
        x, y, headers = load_csv(all_fe_file)
        result["ens_fe"] = y
        result["ens_fe_labels"] = headers

    # Load individual ensemble files
    ens_files = sorted(datadir.glob("ens_[0-9]*_histogram.csv"))
    result["individual_ens"] = {}
    for ef in ens_files:
        # Parse ensemble index from filename like "ens_000_histogram.csv"
        ens_idx_str = ef.stem.split("_")[1]
        try:
            ens_idx = int(ens_idx_str)
        except ValueError:
            continue
        x, y, headers = load_csv(ef)
        # Use histogram column (index 0) for the actual values (time, density, or counts)
        hist = y[:, 0] if y.ndim > 1 else y.flatten()
        fe_file = datadir / f"ens_{ens_idx:03d}_free_energy.csv"
        if fe_file.exists():
            fe_x, fe_y, _ = load_csv(fe_file)
            fe = fe_y[:, 0] if fe_y.ndim > 1 and fe_y.shape[1] > 0 else fe_y.flatten()
        else:
            fe = np.full_like(hist, np.nan)
        result["individual_ens"][ens_idx] = {
            "x": x,
            "hist": hist,
            "fe": fe,
            "label": headers[0] if headers else f"[{ens_idx}+]",
        }

    return result


def plot_wham_results(
    data: Dict,
    interfaces: Optional[List[float]] = None,
    title: str = "",
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot WHAM histogram and free energy side by side.
    Plots [0-] ensemble and WHAM [i+] ensembles as separate lines.

    Parameters
    ----------
    data : dict
        Data dictionary from load_histogram_data.
    interfaces : list of float, optional
        Interface positions to mark.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the figure.
    log_scale : bool
        Use log scale for histogram y-axis. Default is False (linear).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get data - prefer new format, fall back to old
    has_new_format = "wham_plus_x" in data and "ens_0min_x" in data
    
    # Determine y-axis label based on normalization
    normalization = data.get("normalization", "none")
    if normalization == "time":
        ylabel_hist = "Time per dlambda per path"
        title_hist = "WHAM Histogram (Time Normalized)"
    elif normalization == "density":
        ylabel_hist = "Probability density"
        title_hist = "WHAM Histogram (Density Normalized)"
    else:
        ylabel_hist = "Probability P(λ)"
        title_hist = "WHAM Histogram (Probability)"

    # Plot histogram
    ax = axes[0]
    if has_new_format:
        # New format: plot [0-] and WHAM [i+] separately
        x_0min = data.get("ens_0min_x", np.array([]))
        hist_0min = data.get("ens_0min_hist", np.array([]))
        x_plus = data.get("wham_plus_x", np.array([]))
        hist_plus = data.get("wham_plus_hist", np.array([]))

        if len(x_0min) > 0 and len(hist_0min) > 0:
            ax.plot(x_0min, hist_0min, color="tab:green", lw=2, marker="o", markersize=3, label="[0-] ensemble")
        if len(x_plus) > 0 and len(hist_plus) > 0:
            ax.plot(x_plus, hist_plus, color="gold", lw=2, marker="o", markersize=3, label="WHAM [i+] ensembles")
        ax.legend(loc="best")
    else:
        # Old format fallback
        x = data.get("wham_x", np.array([]))
        hist = data.get("wham_hist", np.array([]))
        if len(x) > 0 and len(hist) > 0:
            ax.plot(x, hist, color="tab:blue", lw=2, marker="o", markersize=3)

    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel(ylabel_hist)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title_hist)
    ax.grid(True, alpha=0.3)

    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.3, ls="--")

    # Plot free energy
    ax = axes[1]
    if has_new_format:
        x_0min = data.get("ens_0min_x", np.array([]))
        fe_0min = data.get("ens_0min_fe", np.array([]))
        x_plus = data.get("wham_plus_x", np.array([]))
        fe_plus = data.get("wham_plus_fe", np.array([]))

        if len(x_0min) > 0 and len(fe_0min) > 0:
            ax.plot(x_0min, fe_0min, color="tab:green", lw=2, marker="o", markersize=3, label="[0-] ensemble")
        if len(x_plus) > 0 and len(fe_plus) > 0:
            ax.plot(x_plus, fe_plus, color="gold", lw=2, marker="o", markersize=3, label="WHAM [i+] ensembles")
        ax.legend(loc="best")
    else:
        x = data.get("wham_x", np.array([]))
        fe = data.get("wham_fe", np.array([]))
        if len(x) > 0 and len(fe) > 0:
            ax.plot(x, fe, color="tab:red", lw=2, marker="o", markersize=3)

    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free energy F(λ) [kBT]")
    ax.set_title("Conditional free energy")
    ax.grid(True, alpha=0.3)

    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.3, ls="--")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_wham_vs_0plus(
    data: Dict,
    interfaces: Optional[List[float]] = None,
    title: str = "",
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot overlay of WHAM [i+], [0+] (no WHAM), and [0-] ensembles.
    
    This helps compare the WHAM-weighted result with the raw [0+] ensemble.

    Parameters
    ----------
    data : dict
        Data dictionary from load_histogram_data.
    interfaces : list of float, optional
        Interface positions to mark.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the figure.
    log_scale : bool
        Use log scale for histogram y-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Determine y-axis label based on normalization
    normalization = data.get("normalization", "none")
    if normalization == "time":
        ylabel_hist = "Time per dlambda per path"
        title_hist = "WHAM vs [0+] Histogram (Time Normalized)"
    elif normalization == "density":
        ylabel_hist = "Probability density"
        title_hist = "WHAM vs [0+] Histogram (Density Normalized)"
    else:
        ylabel_hist = "Probability P(λ)"
        title_hist = "WHAM vs [0+] Histogram"

    # Get data
    x_0min = data.get("ens_0min_x", np.array([]))
    hist_0min = data.get("ens_0min_hist", np.array([]))
    fe_0min = data.get("ens_0min_fe", np.array([]))
    
    x_wham = data.get("wham_plus_x", np.array([]))
    hist_wham = data.get("wham_plus_hist", np.array([]))
    fe_wham = data.get("wham_plus_fe", np.array([]))
    
    # Get [0+] ensemble (individual, no WHAM) - ensemble index 1 (since 0 is [0-])
    individual_ens = data.get("individual_ens", {})
    x_0plus = np.array([])
    hist_0plus = np.array([])
    fe_0plus = np.array([])
    if 1 in individual_ens:
        ens_data = individual_ens[1]
        x_0plus = ens_data.get("x", np.array([]))
        hist_0plus = ens_data.get("hist", np.array([]))
        fe_0plus = ens_data.get("fe", np.array([]))

    # Plot histogram
    ax = axes[0]
    if len(x_0min) > 0 and len(hist_0min) > 0:
        ax.plot(x_0min, hist_0min, color="tab:green", lw=2, marker="o", markersize=3, label="[0-] ensemble")
    if len(x_0plus) > 0 and len(hist_0plus) > 0:
        ax.plot(x_0plus, hist_0plus, color="tab:blue", lw=2, marker="s", markersize=3, label="[0+] (no WHAM)")
    if len(x_wham) > 0 and len(hist_wham) > 0:
        ax.plot(x_wham, hist_wham, color="gold", lw=2, marker="o", markersize=3, label="WHAM [i+]")
    
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel(ylabel_hist)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title_hist)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.3, ls="--")

    # Plot free energy
    ax = axes[1]
    if len(x_0min) > 0 and len(fe_0min) > 0:
        ax.plot(x_0min, fe_0min, color="tab:green", lw=2, marker="o", markersize=3, label="[0-] ensemble")
    if len(x_0plus) > 0 and len(fe_0plus) > 0:
        ax.plot(x_0plus, fe_0plus, color="tab:blue", lw=2, marker="s", markersize=3, label="[0+] (no WHAM)")
    if len(x_wham) > 0 and len(fe_wham) > 0:
        ax.plot(x_wham, fe_wham, color="gold", lw=2, marker="o", markersize=3, label="WHAM [i+]")

    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free Energy F(λ) [kBT]")
    ax.set_title("Free Energy: WHAM vs [0+]")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.3, ls="--")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_ensemble_histograms(
    data: Dict,
    interfaces: Optional[List[float]] = None,
    title: str = "",
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot per-ensemble histograms and free energies.

    Parameters
    ----------
    data : dict
        Data dictionary from load_histogram_data.
    interfaces : list of float, optional
        Interface positions to mark.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the figure.
    log_scale : bool
        Use log scale for histogram y-axis. Default is False (linear).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x = data.get("ens_x", np.array([]))
    hist = data.get("ens_hist", np.array([]))
    fe = data.get("ens_fe", np.array([]))
    labels = data.get("ens_labels", [])

    n_ens = hist.shape[1] if hist.ndim > 1 else 1
    
    # Determine y-axis label based on normalization
    normalization = data.get("normalization", "none")
    if normalization == "time":
        ylabel_hist = "Time per bin width"
        title_hist = "Per-Ensemble Histograms (Time Normalized)"
    elif normalization == "density":
        ylabel_hist = "Probability density"
        title_hist = "Per-Ensemble Histograms (Density Normalized)"
    else:
        ylabel_hist = "Probability P(λ)"
        title_hist = "Per-Ensemble Histograms"

    # Plot histograms
    ax = axes[0]
    for i in range(n_ens):
        h = hist[:, i] if hist.ndim > 1 else hist
        label = labels[i] if i < len(labels) else f"Ens {i}"
        color = COLORS[i % len(COLORS)]
        ax.plot(x, h, color=color, lw=2, marker="o", markersize=2, label=label, alpha=0.8)

    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel(ylabel_hist)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title_hist)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    # Plot free energies
    ax = axes[1]
    for i in range(n_ens):
        f = fe[:, i] if fe.ndim > 1 else fe
        label = labels[i] if i < len(labels) else f"Ens {i}"
        color = COLORS[i % len(COLORS)]
        ax.plot(x, f, color=color, lw=1.5, marker="o", markersize=2, label=label, alpha=0.8)

    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free Energy F(λ) [kBT]")
    ax.set_title("Per-Ensemble Free Energies")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_individual_ensemble(
    data: Dict,
    ens_idx: int,
    interfaces: Optional[List[float]] = None,
    title: str = "",
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot histogram and free energy for a single ensemble.

    Parameters
    ----------
    data : dict
        Data dictionary from load_histogram_data.
    ens_idx : int
        Ensemble index.
    interfaces : list of float, optional
        Interface positions to mark.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the figure.
    log_scale : bool
        Use log scale for histogram y-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ens_idx not in data.get("individual_ens", {}):
        print(f"Ensemble {ens_idx} not found in data")
        return None

    ens_data = data["individual_ens"][ens_idx]
    x = ens_data["x"]
    hist = ens_data["hist"]
    fe = ens_data["fe"]
    label = ens_data["label"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax = axes[0]
    ax.plot(x, hist, color="tab:blue", lw=2)
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Probability P(λ)")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Histogram {label}")
    ax.grid(True, alpha=0.3)

    if interfaces:
        for i, intf in enumerate(interfaces):
            alpha = 0.6 if i == ens_idx or i == ens_idx - 1 else 0.2
            ax.axvline(intf, color="k", alpha=alpha, ls="--")

    # Free energy
    ax = axes[1]
    ax.plot(x, fe, color="tab:red", lw=2)
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free Energy F(λ) [kBT]")
    ax.set_title(f"Free Energy {label}")
    ax.grid(True, alpha=0.3)

    if interfaces:
        for i, intf in enumerate(interfaces):
            alpha = 0.6 if i == ens_idx or i == ens_idx - 1 else 0.2
            ax.axvline(intf, color="k", alpha=alpha, ls="--")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def compare_simulations(
    datadirs: List[str],
    labels: Optional[List[str]] = None,
    interfaces: Optional[List[float]] = None,
    title: str = "Simulation Comparison",
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Compare WHAM results from multiple simulations.

    Parameters
    ----------
    datadirs : list of str
        List of data directories.
    labels : list of str, optional
        Labels for each simulation.
    interfaces : list of float, optional
        Interface positions to mark.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the figure.
    log_scale : bool
        Use log scale for histogram y-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if labels is None:
        labels = [f"Sim {i+1}" for i in range(len(datadirs))]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Load all data
    all_data = [load_histogram_data(d) for d in datadirs]

    # Helper to get WHAM data from new or old format
    def get_wham_data(data):
        # Use [i+] ensembles for comparison, fallback to old format
        if "wham_plus_x" in data:
            return data["wham_plus_x"], data.get("wham_plus_hist"), data.get("wham_plus_fe")
        else:
            return data.get("wham_x", np.array([])), data.get("wham_hist"), data.get("wham_fe")

    # WHAM Histogram comparison
    ax = axes[0, 0]
    for i, (data, label) in enumerate(zip(all_data, labels)):
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and hist is not None:
            ax.plot(x, hist, color=COLORS[i % len(COLORS)], lw=2, label=label)
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Probability P(λ)")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("WHAM Histogram Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    # WHAM Free Energy comparison
    ax = axes[0, 1]
    for i, (data, label) in enumerate(zip(all_data, labels)):
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and fe is not None:
            ax.plot(x, fe, color=COLORS[i % len(COLORS)], lw=2, label=label)
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free Energy F(λ) [kBT]")
    ax.set_title("WHAM Free Energy Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    # Free energy difference (relative to first simulation)
    ax = axes[1, 0]
    ref_x, ref_hist, ref_fe = get_wham_data(all_data[0])

    for i, (data, label) in enumerate(zip(all_data[1:], labels[1:]), start=1):
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and fe is not None and ref_fe is not None and len(x) == len(ref_x):
            diff = fe - ref_fe
            ax.plot(x, diff, color=COLORS[i % len(COLORS)], lw=2, label=f"{label} - {labels[0]}")
    ax.axhline(0, color="k", alpha=0.3, ls="-")
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("ΔF [kBT]")
    ax.set_title(f"Free Energy Difference (relative to {labels[0]})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    # Histogram ratio (relative to first simulation)
    ax = axes[1, 1]
    for i, (data, label) in enumerate(zip(all_data[1:], labels[1:]), start=1):
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and hist is not None and ref_hist is not None and len(x) == len(ref_x):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = hist / ref_hist
                ratio[np.isinf(ratio)] = np.nan
            ax.plot(x, ratio, color=COLORS[i % len(COLORS)], lw=2, label=f"{label} / {labels[0]}")
    ax.axhline(1, color="k", alpha=0.3, ls="-")
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("P ratio")
    ax.set_title(f"Histogram Ratio (relative to {labels[0]})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_all(
    datadir: Annotated[str, typer.Option("-datadir", help="Directory with histogram CSV files")] = "histograms",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    title: Annotated[str, typer.Option("-title", help="Plot title")] = "",
    save_dir: Annotated[str, typer.Option("-save", help="Directory to save figures (filenames auto-generated)")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figures")] = True,
    log_scale: Annotated[bool, typer.Option("-log/--linear", help="Use log scale for histograms")] = False,
):
    """
    Plot all histograms and free energies from computed data.
    
    Generates plots showing:
    - WHAM results: [0-] ensemble and WHAM [i+] ensembles as separate lines
    - Combined WHAM (0- + i+) if available
    - Per-ensemble histograms and free energies
    
    Filenames are auto-generated based on path: {system}_{gamma}_subcycles{N}_*.png
    """
    # Parse interfaces
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]

    # Load data
    data = load_histogram_data(datadir)
    
    # Extract system info from path for auto-naming
    system_name, gamma, subcycles = extract_system_info_from_path(datadir)
    
    # Build save paths
    if save_dir:
        wham_filename = build_plot_filename("wham_histogram", system_name, gamma, subcycles)
        ens_filename = build_plot_filename("per_ensemble_histogram", system_name, gamma, subcycles)
        wham_vs_0plus_filename = build_plot_filename("wham_vs_0plus", system_name, gamma, subcycles)
        wham_save = os.path.join(save_dir, wham_filename)
        ens_save = os.path.join(save_dir, ens_filename)
        wham_vs_0plus_save = os.path.join(save_dir, wham_vs_0plus_filename)
    else:
        wham_save = None
        ens_save = None
        wham_vs_0plus_save = None

    # Plot WHAM results ([0-] and [i+] separately)
    plot_wham_results(data, intf_list, title or "WHAM Results", save_path=wham_save, show=show, log_scale=log_scale)

    # Plot WHAM vs [0+] overlay
    plot_wham_vs_0plus(data, intf_list, title or "WHAM vs [0+] Comparison", save_path=wham_vs_0plus_save, show=show, log_scale=log_scale)

    # Plot ensemble results
    if "ens_x" in data:
        plot_ensemble_histograms(data, intf_list, title or "Per-Ensemble Results", save_path=ens_save, show=show, log_scale=log_scale)


def compare(
    datadirs: Annotated[List[str], typer.Argument(help="Directories with histogram CSV files")],
    labels: Annotated[Optional[List[str]], typer.Option("-labels", help="Labels for each simulation")] = None,
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    title: Annotated[str, typer.Option("-title", help="Plot title")] = "Simulation Comparison",
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    log_scale: Annotated[bool, typer.Option("-log/--linear", help="Use log scale for histograms")] = False,
):
    """
    Compare histogram results from multiple simulations.
    """
    # Parse interfaces
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]

    save_path = save if save else None
    compare_simulations(datadirs, labels, intf_list, title, save_path=save_path, show=show, log_scale=log_scale)


def extract_gamma_value(gamma_str: str) -> float:
    """
    Extract numeric gamma value from folder name like 'gamma10' -> 10.0
    
    Parameters
    ----------
    gamma_str : str
        Folder name like 'gamma10', 'gamma0.5', 'gamma100'
    
    Returns
    -------
    float
        Numeric value of gamma
    """
    match = re.search(r"gamma[_-]?(\d+\.?\d*)", gamma_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def find_gamma_directories(parent_dir: str) -> List[Tuple[str, float, str]]:
    """
    Find all gamma* subdirectories with histograms/ folders.
    
    Parameters
    ----------
    parent_dir : str
        Parent directory containing gamma* folders
    
    Returns
    -------
    List[Tuple[str, float, str]]
        List of (histogram_dir_path, gamma_value, folder_name) sorted by gamma value
    """
    parent = Path(parent_dir)
    results = []
    
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower().startswith("gamma"):
            hist_dir = item / "histograms"
            if hist_dir.exists():
                gamma_val = extract_gamma_value(item.name)
                results.append((str(hist_dir), gamma_val, item.name))
    
    # Sort by gamma value
    results.sort(key=lambda x: x[1])
    return results


def extract_subcycles_value(subcycles_str: str) -> int:
    """
    Extract numeric subcycles value from folder name like 'subcycles10' -> 10
    
    Parameters
    ----------
    subcycles_str : str
        Folder name like 'subcycles10', 'subcycle1', 'subcycles100'
    
    Returns
    -------
    int
        Numeric value of subcycles
    """
    match = re.search(r"subcycles?[_-]?(\d+)", subcycles_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def find_subcycles_directories(parent_dir: str) -> List[Tuple[str, int, str]]:
    """
    Find all subcycle* subdirectories with histograms/ folders.
    
    Parameters
    ----------
    parent_dir : str
        Parent directory containing subcycle* folders
    
    Returns
    -------
    List[Tuple[str, int, str]]
        List of (histogram_dir_path, subcycles_value, folder_name) sorted by subcycles value
    """
    parent = Path(parent_dir)
    results = []
    
    for item in parent.iterdir():
        if item.is_dir() and item.name.lower().startswith("subcycle"):
            hist_dir = item / "histograms"
            if hist_dir.exists():
                subcycles_val = extract_subcycles_value(item.name)
                results.append((str(hist_dir), subcycles_val, item.name))
    
    # Sort by subcycles value
    results.sort(key=lambda x: x[1])
    return results


def extract_mass_value(mass_str: str) -> float:
    """
    Extract numeric mass value from folder name like 'mass0.05' -> 0.05
    """
    match = re.search(r"mass[_-]?(\d+\.?\d*)", mass_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def find_mass_directories(parent_dir: str) -> List[Tuple[str, float, str]]:
    """
    Find all mass* subdirectories with histograms/ folders.

    Returns list of (histogram_dir_path, mass_value, folder_name) sorted by mass value
    """
    parent = Path(parent_dir)
    results = []

    for item in parent.iterdir():
        if item.is_dir() and item.name.lower().startswith("mass"):
            hist_dir = item / "histograms"
            if hist_dir.exists():
                mass_val = extract_mass_value(item.name)
                results.append((str(hist_dir), mass_val, item.name))

    results.sort(key=lambda x: x[1])
    return results


def get_gamma_colormap(values: List[float], cmap_name: str = "viridis", vmin: float = 0.0, vmax: float = 1.0) -> Dict[float, np.ndarray]:
    """
    Create a colormap with uniformly spaced colors for a list of values.
    
    Parameters
    ----------
    values : list of float
        List of values (gamma, subcycles, etc.)
    cmap_name : str
        Matplotlib colormap name (default: viridis)
    vmin : float
        Minimum normalized value for colormap (default: 0.0)
    vmax : float
        Maximum normalized value for colormap (default: 1.0)
    
    Returns
    -------
    dict
        Mapping from value to RGBA color
    """
    n = len(values)
    if n == 1:
        cmap = cm.get_cmap(cmap_name)
        return {values[0]: cmap(0.5 * (vmin + vmax))}
    
    cmap = cm.get_cmap(cmap_name)
    
    # Use uniformly spaced colors (not scaled by value)
    colors = {}
    for i, v in enumerate(values):
        norm_val = vmin + (vmax - vmin) * i / (n - 1)  # Uniform spacing from vmin to vmax
        colors[v] = cmap(norm_val)
    
    return colors


def compare_gamma_simulations(
    parent_dir: str = None,
    histogram_dirs: List[str] = None,
    interfaces: Optional[List[float]] = None,
    title: str = "Gamma Comparison",
    figsize: Tuple[float, float] = (16, 5),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
    cmap: str = "viridis",
    plot_0minus: bool = True,
    plot_0plus: bool = False,
) -> plt.Figure:
    """
    Compare WHAM histograms from simulations with different gamma values.
    
    Uses a colormap based on gamma values for intuitive visualization.
    
    Parameters
    ----------
    parent_dir : str, optional
        Parent directory containing gamma* subfolders, each with histograms/
    histogram_dirs : list of str, optional
        Explicit list of histogram directories (alternative to parent_dir)
    interfaces : list of float, optional
        Interface positions to mark
    title : str
        Figure title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    log_scale : bool
        Use log scale for histogram y-axis
    cmap : str
        Matplotlib colormap name (default: viridis)
    plot_0minus : bool
        Include [0-] ensemble histogram in plots (default: True)
    plot_0plus : bool
        Include [0+] (no WHAM) ensemble histogram in plots (default: False)
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Find gamma directories
    if parent_dir:
        gamma_dirs = find_gamma_directories(parent_dir)
        if not gamma_dirs:
            raise ValueError(f"No gamma*/histograms directories found in {parent_dir}")
        
        dirs = [d[0] for d in gamma_dirs]
        gamma_values = [d[1] for d in gamma_dirs]
        gamma_names = [d[2] for d in gamma_dirs]
    elif histogram_dirs:
        dirs = histogram_dirs
        gamma_values = []
        gamma_names = []
        for d in dirs:
            _, gamma_name, _ = extract_system_info_from_path(d)
            if gamma_name:
                gamma_values.append(extract_gamma_value(gamma_name))
                gamma_names.append(gamma_name)
            else:
                gamma_values.append(0.0)
                gamma_names.append(Path(d).parent.name)
    else:
        raise ValueError("Must provide either parent_dir or histogram_dirs")
    
    # Create color mapping
    colors = get_gamma_colormap(gamma_values, cmap)
    
    # Load all data
    all_data = []
    for d in dirs:
        try:
            data = load_histogram_data(d)
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load data from {d}: {e}")
            all_data.append({})
    
    # Create figure with 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Helper to get WHAM data
    def get_wham_data(data):
        if "wham_plus_x" in data:
            return data["wham_plus_x"], data.get("wham_plus_hist"), data.get("wham_plus_fe")
        return data.get("wham_x", np.array([])), data.get("wham_hist"), data.get("wham_fe")
    
    def get_0minus_data(data):
        return data.get("ens_0min_x", np.array([])), data.get("ens_0min_hist"), data.get("ens_0min_fe")
    
    def get_0plus_data(data):
        individual_ens = data.get("individual_ens", {})
        if 1 in individual_ens:
            ens_data = individual_ens[1]
            return ens_data.get("x", np.array([])), ens_data.get("hist"), ens_data.get("fe")
        return np.array([]), None, None
    
    # Determine y-axis label based on normalization of first valid dataset
    normalization = "none"
    for data in all_data:
        if data:
            normalization = data.get("normalization", "none")
            break
    
    if normalization == "time":
        ylabel_hist = "Time per dlambda per path"
    elif normalization == "density":
        ylabel_hist = "Probability density"
    else:
        ylabel_hist = "Probability P(λ)"
    
    # ===== Plot 1: WHAM [i+] Histogram =====
    ax = axes[0]
    for data, gval, gname in zip(all_data, gamma_values, gamma_names):
        if not data:
            continue
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and hist is not None:
            ax.plot(x, hist, color=colors[gval], lw=2, label=f"γ={gval:.0f}")
    
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel(ylabel_hist)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("WHAM [i+] Histogram")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    # ===== Plot 2: WHAM [i+] Free Energy =====
    ax = axes[1]
    for data, gval, gname in zip(all_data, gamma_values, gamma_names):
        if not data:
            continue
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and fe is not None:
            ax.plot(x, fe, color=colors[gval], lw=2, label=f"γ={gval:.0f}")
    
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free Energy F(λ) [kBT]")
    ax.set_title("WHAM [i+] Free Energy")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    # ===== Plot 3: [0-] or [0+] Histogram =====
    ax = axes[2]
    if plot_0minus:
        for data, gval, gname in zip(all_data, gamma_values, gamma_names):
            if not data:
                continue
            x, hist, fe = get_0minus_data(data)
            if x is not None and len(x) > 0 and hist is not None:
                ax.plot(x, hist, color=colors[gval], lw=2, label=f"γ={gval:.0f}")
        ax.set_title("[0-] Ensemble Histogram")
    elif plot_0plus:
        for data, gval, gname in zip(all_data, gamma_values, gamma_names):
            if not data:
                continue
            x, hist, fe = get_0plus_data(data)
            if x is not None and len(x) > 0 and hist is not None:
                ax.plot(x, hist, color=colors[gval], lw=2, label=f"γ={gval:.0f}")
        ax.set_title("[0+] Ensemble Histogram (no WHAM)")
    else:
        ax.set_visible(False)
    
    if ax.get_visible():
        ax.set_xlabel("Order parameter λ")
        ax.set_ylabel(ylabel_hist)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        if interfaces:
            for intf in interfaces:
                ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        # Save both PNG and PDF
        base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
        png_path = base_path + '.png'
        pdf_path = base_path + '.pdf'
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    
    if show:
        plt.show()
    
    # === Create second figure: individual ensembles + WHAM vs 0+ comparison ===
    fig2 = plot_ensemble_comparison(
        all_data=all_data,
        values=gamma_values,
        names=gamma_names,
        colors=colors,
        interfaces=interfaces,
        title=f"{title} - Ensemble Comparison",
        ylabel_hist=ylabel_hist,
        log_scale=log_scale,
        label_prefix="γ",
        save_path=save_path.rsplit('.', 1)[0] + '_ensembles' if save_path and '.' in save_path else (save_path + '_ensembles' if save_path else None),
        show=show,
    )
    # === Create third figure: symmetrized WHAM free energies ===
    try:
        # Collect WHAM free energies and x-values
        fe_list = []
        x_common = None
        for data, gval, gname in zip(all_data, gamma_values, gamma_names):
            if not data:
                continue
            xw, histw, few = get_wham_data(data)
            if xw is None or few is None:
                continue
            if x_common is None:
                x_common = xw
            # Ensure x arrays match; if not, interpolate onto x_common later
            fe_list.append((gval, xw, few))

        if fe_list and x_common is not None:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            xmin = x_common[0]
            xmax = x_common[-1]
            for gval, xw, few in fe_list:
                # Interpolate fe onto x_common if needed
                if not np.array_equal(xw, x_common):
                    fe_interp = np.interp(x_common, xw, few, left=np.nan, right=np.nan)
                else:
                    fe_interp = few

                # Mirror and average around 0: fe_sym(x) = mean( fe(x), fe(-x) )
                x_mirror = -x_common
                # Interpolate mirrored values onto the common grid; out-of-range gives NaN
                f_mirror = np.interp(x_mirror, x_common, fe_interp, left=np.nan, right=np.nan)

                fe_stack = np.vstack([fe_interp, f_mirror])
                # Suppress warnings when taking nanmean over all-NaN slices
                with np.errstate(invalid='ignore'):
                    fe_sym = np.nanmean(fe_stack, axis=0)

                # Shift to min=0 for plotting clarity
                if np.all(np.isnan(fe_sym)):
                    continue
                fe_sym = fe_sym - np.nanmin(fe_sym)
                ax3.plot(x_common, fe_sym, color=colors[gval], lw=2, label=f"γ={gval:.0f}")

            ax3.set_xlabel("Order parameter λ")
            ax3.set_ylabel("Symmetrized Free Energy F(λ) [kBT]")
            ax3.set_title(f"{title} - Symmetrized WHAM Free Energies")
            ax3.legend(loc="best", fontsize=9)
            ax3.grid(True, alpha=0.3)

            if save_path:
                base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
                png_path = base_path + '_fe_symm.png'
                pdf_path = base_path + '_fe_symm.pdf'
                plt.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
                print(f"Saved: {png_path}")
                print(f"Saved: {pdf_path}")
            if show:
                plt.show()
    except Exception as e:
        print(f"Warning: could not create symmetrized FE figure: {e}")
    
    # === Create delta_F vs gamma and save CSV ===
    try:
        delta_vals = []
        params = []
        for data, gval, gname in zip(all_data, gamma_values, gamma_names):
            if not data:
                params.append(gval)
                delta_vals.append(np.nan)
                continue
            xw, histw, few = get_wham_data(data)
            fe_arr = None
            if few is not None:
                fe_arr = np.asarray(few)
            elif "wham_plus_fe" in data:
                fe_arr = np.asarray(data.get("wham_plus_fe"))

            if fe_arr is None:
                params.append(gval)
                delta_vals.append(np.nan)
                continue

            valid = np.where(np.isfinite(fe_arr))[0]
            if valid.size > 0:
                delta = float(fe_arr[valid[-1]])
            else:
                delta = np.nan
            params.append(gval)
            delta_vals.append(delta)

        # Plot and save
        if any(np.isfinite(delta_vals)):
            figd, axd = plt.subplots(figsize=(6, 4))
            axd.plot(params, delta_vals, 'o-', color='tab:blue')
            axd.set_xlabel('gamma')
            axd.set_ylabel('Delta F (final FE value) [kBT]')
            axd.set_title(f'{title} - DeltaF vs gamma')
            axd.grid(True, alpha=0.3)
            if save_path:
                base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
                png_path = base_path + '_deltaF.png'
                pdf_path = base_path + '_deltaF.pdf'
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {png_path}")
                print(f"Saved: {pdf_path}")
            # Save CSV
            if save_path:
                csv_path = base_path + '_deltaF.csv'
                with open(csv_path, 'w') as fh:
                    fh.write('gamma,delta_F\n')
                    for p, d in zip(params, delta_vals):
                        fh.write(f"{p},{d}\n")
                print(f"Saved: {csv_path}")
            if show:
                plt.show()
    except Exception as e:
        print(f"Warning: could not create deltaF figure: {e}")

    return fig


def plot_ensemble_comparison(
    all_data: List[Dict],
    values: List[float],
    names: List[str],
    colors: Dict[float, np.ndarray],
    interfaces: Optional[List[float]] = None,
    title: str = "Ensemble Comparison",
    ylabel_hist: str = "Probability P(λ)",
    log_scale: bool = False,
    label_prefix: str = "γ",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a figure comparing individual (non-WHAM) ensembles across simulations.
    
    Creates panels for each ensemble [0+], [1+], [2+], etc., plus a panel
    comparing WHAM vs [0+] for each simulation.
    
    Parameters
    ----------
    all_data : list of dict
        Data dictionaries from load_histogram_data
    values : list of float
        Parameter values (gamma or subcycles)
    names : list of str
        Parameter names for labels
    colors : dict
        Mapping from value to color
    interfaces : list of float, optional
        Interface positions to mark
    title : str
        Figure title
    ylabel_hist : str
        Y-axis label for histograms
    log_scale : bool
        Use log scale for y-axis
    label_prefix : str
        Prefix for legend labels (e.g., "γ" or "sc")
    save_path : str, optional
        Path to save figure (without extension)
    show : bool
        Whether to display the figure
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Find all available ensemble indices across all datasets
    all_ens_indices = set()
    for data in all_data:
        if data and "individual_ens" in data:
            all_ens_indices.update(data["individual_ens"].keys())
    
    if not all_ens_indices:
        print("Warning: No individual ensemble data found")
        return None
    
    # Sort ensemble indices (excluding 0 which is [0-])
    ens_indices = sorted([i for i in all_ens_indices if i > 0])
    
    # Number of panels: one per ensemble + one for WHAM vs [0+] comparison
    n_panels = len(ens_indices) + 1
    
    # Determine grid layout (try to be roughly square)
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Helper to get ensemble data
    def get_ensemble_data(data, ens_idx):
        individual_ens = data.get("individual_ens", {})
        if ens_idx in individual_ens:
            ens_data = individual_ens[ens_idx]
            return ens_data.get("x", np.array([])), ens_data.get("hist"), ens_data.get("label", f"[{ens_idx}+]")
        return np.array([]), None, f"[{ens_idx}+]"
    
    def get_wham_data(data):
        if "wham_plus_x" in data:
            return data["wham_plus_x"], data.get("wham_plus_hist")
        return data.get("wham_x", np.array([])), data.get("wham_hist")
    
    # Plot each ensemble in its own panel
    for panel_idx, ens_idx in enumerate(ens_indices):
        ax = axes[panel_idx]
        
        for data, val, name in zip(all_data, values, names):
            if not data:
                continue
            x, hist, label = get_ensemble_data(data, ens_idx)
            if x is not None and len(x) > 0 and hist is not None:
                ax.plot(x, hist, color=colors[val], lw=2, label=f"{label_prefix}={val:.0f}")
        
        ax.set_xlabel("Order parameter λ")
        ax.set_ylabel(ylabel_hist)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"[{ens_idx-1}+] Ensemble (no WHAM)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        if interfaces:
            for intf in interfaces:
                ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    # Final panel: WHAM vs [0+] comparison
    ax = axes[len(ens_indices)]
    
    for data, val, name in zip(all_data, values, names):
        if not data:
            continue
        # Plot WHAM with solid line
        x_wham, hist_wham = get_wham_data(data)
        if x_wham is not None and len(x_wham) > 0 and hist_wham is not None:
            ax.plot(x_wham, hist_wham, color=colors[val], lw=2, ls='-', 
                   label=f"WHAM {label_prefix}={val:.0f}")
        
        # Plot [0+] with dashed line (ens_idx=1 corresponds to [0+])
        x_0plus, hist_0plus, _ = get_ensemble_data(data, 1)
        if x_0plus is not None and len(x_0plus) > 0 and hist_0plus is not None:
            ax.plot(x_0plus, hist_0plus, color=colors[val], lw=2, ls='--',
                   label=f"[0+] {label_prefix}={val:.0f}")
    
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel(ylabel_hist)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("WHAM [i+] vs [0+] (solid vs dashed)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    # Hide unused axes
    for idx in range(len(ens_indices) + 1, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
        png_path = base_path + '.png'
        pdf_path = base_path + '.pdf'
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    
    if show:
        plt.show()
    
    return fig


def compare_gamma(
    parent_dir: Annotated[str, typer.Argument(help="Parent directory containing gamma* folders with histograms/")] = ".",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    title: Annotated[str, typer.Option("-title", help="Plot title")] = "Gamma Comparison",
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    log_scale: Annotated[bool, typer.Option("-log/--linear", help="Use log scale for histograms")] = False,
    cmap: Annotated[str, typer.Option("-cmap", help="Colormap name (viridis, plasma, coolwarm, etc.)")] = "viridis",
    plot_0minus: Annotated[bool, typer.Option("-0minus/--no-0minus", help="Include [0-] ensemble histogram")] = True,
    plot_0plus: Annotated[bool, typer.Option("-0plus/--no-0plus", help="Include [0+] (no WHAM) ensemble histogram")] = False,
):
    """
    Compare histograms across different gamma values.
    
    Automatically finds gamma* subdirectories with histograms/ folders and
    plots them with a color scale based on gamma value.
    
    Example:
        inft compare_gamma /path/to/system -save gamma_comparison.png
        inft compare_gamma . -cmap plasma -log
    """
    # Parse interfaces
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]
    
    save_path = save if save else None
    
    compare_gamma_simulations(
        parent_dir=parent_dir,
        interfaces=intf_list,
        title=title,
        save_path=save_path,
        show=show,
        log_scale=log_scale,
        cmap=cmap,
        plot_0minus=plot_0minus,
        plot_0plus=plot_0plus,
    )


def compare_subcycles_simulations(
    parent_dir: str = None,
    histogram_dirs: List[str] = None,
    interfaces: Optional[List[float]] = None,
    title: str = "Subcycles Comparison",
    figsize: Tuple[float, float] = (16, 5),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
    cmap: str = "plasma",
    plot_0minus: bool = True,
    plot_0plus: bool = False,
) -> plt.Figure:
    """
    Compare WHAM histograms from simulations with different subcycles values.
    
    Uses a colormap based on subcycles values for intuitive visualization.
    
    Parameters
    ----------
    parent_dir : str, optional
        Parent directory containing subcycle* subfolders, each with histograms/
    histogram_dirs : list of str, optional
        Explicit list of histogram directories (alternative to parent_dir)
    interfaces : list of float, optional
        Interface positions to mark
    title : str
        Figure title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    log_scale : bool
        Use log scale for histogram y-axis
    cmap : str
        Matplotlib colormap name (default: plasma)
    plot_0minus : bool
        Include [0-] ensemble histogram in plots (default: True)
    plot_0plus : bool
        Include [0+] (no WHAM) ensemble histogram in plots (default: False)
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Find subcycles directories
    if parent_dir:
        subcycles_dirs = find_subcycles_directories(parent_dir)
        if not subcycles_dirs:
            raise ValueError(f"No subcycle*/histograms directories found in {parent_dir}")
        
        dirs = [d[0] for d in subcycles_dirs]
        subcycles_values = [float(d[1]) for d in subcycles_dirs]  # Convert to float for colormap
        subcycles_names = [d[2] for d in subcycles_dirs]
    elif histogram_dirs:
        dirs = histogram_dirs
        subcycles_values = []
        subcycles_names = []
        for d in dirs:
            folder_name = Path(d).parent.name
            subcycles_values.append(float(extract_subcycles_value(folder_name)))
            subcycles_names.append(folder_name)
    else:
        raise ValueError("Must provide either parent_dir or histogram_dirs")
    
    # Create color mapping
    colors = get_gamma_colormap(subcycles_values, cmap)
    
    # Load all data
    all_data = []
    for d in dirs:
        try:
            data = load_histogram_data(d)
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load data from {d}: {e}")
            all_data.append({})
    
    # Create figure with 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Helper to get WHAM data
    def get_wham_data(data):
        if "wham_plus_x" in data:
            return data["wham_plus_x"], data.get("wham_plus_hist"), data.get("wham_plus_fe")
        return data.get("wham_x", np.array([])), data.get("wham_hist"), data.get("wham_fe")
    
    def get_0minus_data(data):
        return data.get("ens_0min_x", np.array([])), data.get("ens_0min_hist"), data.get("ens_0min_fe")
    
    def get_0plus_data(data):
        individual_ens = data.get("individual_ens", {})
        if 1 in individual_ens:
            ens_data = individual_ens[1]
            return ens_data.get("x", np.array([])), ens_data.get("hist"), ens_data.get("fe")
        return np.array([]), None, None
    
    # Determine y-axis label based on normalization of first valid dataset
    normalization = "none"
    for data in all_data:
        if data:
            normalization = data.get("normalization", "none")
            break
    
    if normalization == "time":
        ylabel_hist = "Time per dlambda per path"
    elif normalization == "density":
        ylabel_hist = "Probability density"
    else:
        ylabel_hist = "Probability P(λ)"
    
    # ===== Plot 1: WHAM [i+] Histogram =====
    ax = axes[0]
    for data, sval, sname in zip(all_data, subcycles_values, subcycles_names):
        if not data:
            continue
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and hist is not None:
            ax.plot(x, hist, color=colors[sval], lw=2, label=f"sc={int(sval)}")
    
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel(ylabel_hist)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("WHAM [i+] Histogram")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    # ===== Plot 2: WHAM [i+] Free Energy =====
    ax = axes[1]
    for data, sval, sname in zip(all_data, subcycles_values, subcycles_names):
        if not data:
            continue
        x, hist, fe = get_wham_data(data)
        if x is not None and len(x) > 0 and fe is not None:
            ax.plot(x, fe, color=colors[sval], lw=2, label=f"sc={int(sval)}")
    
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free Energy F(λ) [kBT]")
    ax.set_title("WHAM [i+] Free Energy")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    # ===== Plot 3: [0-] or [0+] Histogram =====
    ax = axes[2]
    if plot_0minus:
        for data, sval, sname in zip(all_data, subcycles_values, subcycles_names):
            if not data:
                continue
            x, hist, fe = get_0minus_data(data)
            if x is not None and len(x) > 0 and hist is not None:
                ax.plot(x, hist, color=colors[sval], lw=2, label=f"sc={int(sval)}")
        ax.set_title("[0-] Ensemble Histogram")
    elif plot_0plus:
        for data, sval, sname in zip(all_data, subcycles_values, subcycles_names):
            if not data:
                continue
            x, hist, fe = get_0plus_data(data)
            if x is not None and len(x) > 0 and hist is not None:
                ax.plot(x, hist, color=colors[sval], lw=2, label=f"sc={int(sval)}")
        ax.set_title("[0+] Ensemble Histogram (no WHAM)")
    else:
        ax.set_visible(False)
    
    if ax.get_visible():
        ax.set_xlabel("Order parameter λ")
        ax.set_ylabel(ylabel_hist)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        if interfaces:
            for intf in interfaces:
                ax.axvline(intf, color="k", alpha=0.2, ls="--")
    
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        # Save both PNG and PDF
        base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
        png_path = base_path + '.png'
        pdf_path = base_path + '.pdf'
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    
    if show:
        plt.show()
    
    # === Create second figure: individual ensembles + WHAM vs 0+ comparison ===
    fig2 = plot_ensemble_comparison(
        all_data=all_data,
        values=subcycles_values,
        names=subcycles_names,
        colors=colors,
        interfaces=interfaces,
        title=f"{title} - Ensemble Comparison",
        ylabel_hist=ylabel_hist,
        log_scale=log_scale,
        label_prefix="sc",
        save_path=save_path.rsplit('.', 1)[0] + '_ensembles' if save_path and '.' in save_path else (save_path + '_ensembles' if save_path else None),
        show=show,
    )
    # === Create symmetrized WHAM free energies (per subcycle) ===
    try:
        fe_list = []
        x_common = None
        for data, sval, sname in zip(all_data, subcycles_values, subcycles_names):
            if not data:
                continue
            xw, histw, few = get_wham_data(data)
            if xw is None or few is None:
                continue
            if x_common is None:
                x_common = xw
            fe_list.append((sval, xw, few))

        if fe_list and x_common is not None:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            for sval, xw, few in fe_list:
                if not np.array_equal(xw, x_common):
                    fe_interp = np.interp(x_common, xw, few, left=np.nan, right=np.nan)
                else:
                    fe_interp = few

                # Mirror and average around 0: fe_sym(x) = mean( fe(x), fe(-x) )
                x_mirror = -x_common
                f_mirror = np.interp(x_mirror, x_common, fe_interp, left=np.nan, right=np.nan)

                fe_stack = np.vstack([fe_interp, f_mirror])
                with np.errstate(invalid='ignore'):
                    fe_sym = np.nanmean(fe_stack, axis=0)

                if np.all(np.isnan(fe_sym)):
                    continue
                fe_sym = fe_sym - np.nanmin(fe_sym)
                ax3.plot(x_common, fe_sym, color=colors[sval], lw=2, label=f"sc={int(sval)}")

            ax3.set_xlabel("Order parameter λ")
            ax3.set_ylabel("Symmetrized Free Energy F(λ) [kBT]")
            ax3.set_title(f"{title} - Symmetrized WHAM Free Energies")
            ax3.legend(loc="best", fontsize=9)
            ax3.grid(True, alpha=0.3)

            if save_path:
                base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
                png_path = base_path + '_fe_symm.png'
                pdf_path = base_path + '_fe_symm.pdf'
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {png_path}")
                print(f"Saved: {pdf_path}")
            if show:
                plt.show()
    except Exception as e:
        print(f"Warning: could not create symmetrized FE figure: {e}")
    
    # === Create delta_F vs subcycles and save CSV ===
    try:
        delta_vals = []
        params = []
        for data, sval, sname in zip(all_data, subcycles_values, subcycles_names):
            if not data:
                params.append(sval)
                delta_vals.append(np.nan)
                continue
            xw, histw, few = get_wham_data(data)
            fe_arr = None
            if few is not None:
                fe_arr = np.asarray(few)
            elif "wham_plus_fe" in data:
                fe_arr = np.asarray(data.get("wham_plus_fe"))

            if fe_arr is None:
                params.append(sval)
                delta_vals.append(np.nan)
                continue

            valid = np.where(np.isfinite(fe_arr))[0]
            if valid.size > 0:
                delta = float(fe_arr[valid[-1]])
            else:
                delta = np.nan
            params.append(sval)
            delta_vals.append(delta)

        # Plot and save
        if any(np.isfinite(delta_vals)):
            figd, axd = plt.subplots(figsize=(6, 4))
            axd.plot(params, delta_vals, 'o-', color='tab:blue')
            axd.set_xlabel('subcycles')
            axd.set_ylabel('Delta F (final FE value) [kBT]')
            axd.set_title(f'{title} - DeltaF vs subcycles')
            axd.grid(True, alpha=0.3)
            if save_path:
                base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
                png_path = base_path + '_deltaF.png'
                pdf_path = base_path + '_deltaF.pdf'
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {png_path}")
                print(f"Saved: {pdf_path}")
            # Save CSV
            if save_path:
                csv_path = base_path + '_deltaF.csv'
                with open(csv_path, 'w') as fh:
                    fh.write('subcycles,delta_F\n')
                    for p, d in zip(params, delta_vals):
                        fh.write(f"{p},{d}\n")
                print(f"Saved: {csv_path}")
            if show:
                plt.show()
    except Exception as e:
        print(f"Warning: could not create deltaF figure: {e}")

    return fig


def compare_subcycles(
    parent_dir: Annotated[str, typer.Argument(help="Parent directory containing subcycle* folders with histograms/")] = ".",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    title: Annotated[str, typer.Option("-title", help="Plot title")] = "Subcycles Comparison",
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    log_scale: Annotated[bool, typer.Option("-log/--linear", help="Use log scale for histograms")] = False,
    cmap: Annotated[str, typer.Option("-cmap", help="Colormap name (viridis, plasma, coolwarm, etc.)")] = "plasma",
    plot_0minus: Annotated[bool, typer.Option("-0minus/--no-0minus", help="Include [0-] ensemble histogram")] = True,
    plot_0plus: Annotated[bool, typer.Option("-0plus/--no-0plus", help="Include [0+] (no WHAM) ensemble histogram")] = False,
):
    """
    Compare histograms across different subcycles values.
    
    Automatically finds subcycle* subdirectories with histograms/ folders and
    plots them with a color scale based on subcycles value.
    
    Example:
        inft compare_subcycles /path/to/system -save subcycles_comparison.png
        inft compare_subcycles . -cmap viridis -log
    """
    # Parse interfaces
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]
    
    save_path = save if save else None
    
    compare_subcycles_simulations(
        parent_dir=parent_dir,
        interfaces=intf_list,
        title=title,
        save_path=save_path,
        show=show,
        log_scale=log_scale,
        cmap=cmap,
        plot_0minus=plot_0minus,
        plot_0plus=plot_0plus,
    )


def compare_mass_simulations(
    parent_dir: str = None,
    histogram_dirs: List[str] = None,
    interfaces: Optional[List[float]] = None,
    title: str = "Mass Comparison",
    figsize: Tuple[float, float] = (16, 5),
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False,
    cmap: str = "plasma",
    plot_0minus: bool = True,
    plot_0plus: bool = False,
) -> plt.Figure:
    """
    Compare WHAM histograms from simulations with different mass values.

    Mirrors the behaviour of `compare_gamma_simulations` and
    `compare_subcycles_simulations` (same plots: hist, FE, 0- / 0+; symmetrized FE; deltaF CSV/plot).
    """
    # Find mass directories
    if parent_dir:
        mass_dirs = find_mass_directories(parent_dir)
        if not mass_dirs:
            raise ValueError(f"No mass*/histograms directories found in {parent_dir}")

        dirs = [d[0] for d in mass_dirs]
        mass_values = [d[1] for d in mass_dirs]
        mass_names = [d[2] for d in mass_dirs]
    elif histogram_dirs:
        dirs = histogram_dirs
        mass_values = []
        mass_names = []
        for d in dirs:
            folder_name = Path(d).parent.name
            mass_values.append(float(extract_mass_value(folder_name)))
            mass_names.append(folder_name)
    else:
        raise ValueError("Must provide either parent_dir or histogram_dirs")

    colors = get_gamma_colormap(mass_values, cmap)

    all_data = []
    for d in dirs:
        try:
            data = load_histogram_data(d)
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load data from {d}: {e}")
            all_data.append({})

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    def get_wham_data_local(data):
        if "wham_plus_x" in data:
            return data["wham_plus_x"], data.get("wham_plus_hist"), data.get("wham_plus_fe")
        return data.get("wham_x", np.array([])), data.get("wham_hist"), data.get("wham_fe")

    def get_0minus_data_local(data):
        return data.get("ens_0min_x", np.array([])), data.get("ens_0min_hist"), data.get("ens_0min_fe")

    def get_0plus_data_local(data):
        individual_ens = data.get("individual_ens", {})
        if 1 in individual_ens:
            ens_data = individual_ens[1]
            return ens_data.get("x", np.array([])), ens_data.get("hist"), ens_data.get("fe")
        return np.array([]), None, None

    # Determine normalization label
    normalization = "none"
    for data in all_data:
        if data:
            normalization = data.get("normalization", "none")
            break

    if normalization == "time":
        ylabel_hist = "Time per dlambda per path"
    elif normalization == "density":
        ylabel_hist = "Probability density"
    else:
        ylabel_hist = "Probability P(λ)"

    # Plot 1: Histogram
    ax = axes[0]
    for data, mval, mname in zip(all_data, mass_values, mass_names):
        if not data:
            continue
        x, hist, fe = get_wham_data_local(data)
        if x is not None and len(x) > 0 and hist is not None:
            ax.plot(x, hist, color=colors[mval], lw=2, label=f"m={mval}")
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel(ylabel_hist)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("WHAM [i+] Histogram")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    # Plot 2: Free energy
    ax = axes[1]
    for data, mval, mname in zip(all_data, mass_values, mass_names):
        if not data:
            continue
        x, hist, fe = get_wham_data_local(data)
        if x is not None and len(x) > 0 and fe is not None:
            ax.plot(x, fe, color=colors[mval], lw=2, label=f"m={mval}")
    ax.set_xlabel("Order parameter λ")
    ax.set_ylabel("Free Energy F(λ) [kBT]")
    ax.set_title("WHAM [i+] Free Energy")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if interfaces:
        for intf in interfaces:
            ax.axvline(intf, color="k", alpha=0.2, ls="--")

    # Plot 3: 0- or 0+
    ax = axes[2]
    if plot_0minus:
        for data, mval, mname in zip(all_data, mass_values, mass_names):
            if not data:
                continue
            x, hist, fe = get_0minus_data_local(data)
            if x is not None and len(x) > 0 and hist is not None:
                ax.plot(x, hist, color=colors[mval], lw=2, label=f"m={mval}")
        ax.set_title("[0-] Ensemble Histogram")
    elif plot_0plus:
        for data, mval, mname in zip(all_data, mass_values, mass_names):
            if not data:
                continue
            x, hist, fe = get_0plus_data_local(data)
            if x is not None and len(x) > 0 and hist is not None:
                ax.plot(x, hist, color=colors[mval], lw=2, label=f"m={mval}")
        ax.set_title("[0+] Ensemble Histogram (no WHAM)")
    else:
        ax.set_visible(False)

    if ax.get_visible():
        ax.set_xlabel("Order parameter λ")
        ax.set_ylabel(ylabel_hist)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        if interfaces:
            for intf in interfaces:
                ax.axvline(intf, color="k", alpha=0.2, ls="--")

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
        png_path = base_path + '.png'
        pdf_path = base_path + '.pdf'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

    if show:
        plt.show()

    # === ensemble comparison figure ===
    fig2 = plot_ensemble_comparison(
        all_data=all_data,
        values=mass_values,
        names=mass_names,
        colors=colors,
        interfaces=interfaces,
        title=f"{title} - Ensemble Comparison",
        ylabel_hist=ylabel_hist,
        log_scale=log_scale,
        label_prefix="m",
        save_path=save_path.rsplit('.', 1)[0] + '_ensembles' if save_path and '.' in save_path else (save_path + '_ensembles' if save_path else None),
        show=show,
    )

    # === symmetrized FE per mass ===
    try:
        fe_list = []
        x_common = None
        for data, mval, mname in zip(all_data, mass_values, mass_names):
            if not data:
                continue
            xw, histw, few = get_wham_data_local(data)
            if xw is None or few is None:
                continue
            if x_common is None:
                x_common = xw
            fe_list.append((mval, xw, few))

        if fe_list and x_common is not None:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            for mval, xw, few in fe_list:
                if not np.array_equal(xw, x_common):
                    fe_interp = np.interp(x_common, xw, few, left=np.nan, right=np.nan)
                else:
                    fe_interp = few

                x_mirror = -x_common
                f_mirror = np.interp(x_mirror, x_common, fe_interp, left=np.nan, right=np.nan)
                fe_stack = np.vstack([fe_interp, f_mirror])
                with np.errstate(invalid='ignore'):
                    fe_sym = np.nanmean(fe_stack, axis=0)

                if np.all(np.isnan(fe_sym)):
                    continue
                fe_sym = fe_sym - np.nanmin(fe_sym)
                ax3.plot(x_common, fe_sym, color=colors[mval], lw=2, label=f"m={mval}")

            ax3.set_xlabel("Order parameter λ")
            ax3.set_ylabel("Symmetrized Free Energy F(λ) [kBT]")
            ax3.set_title(f"{title} - Symmetrized WHAM Free Energies")
            ax3.legend(loc="best", fontsize=9)
            ax3.grid(True, alpha=0.3)

            if save_path:
                base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
                png_path = base_path + '_fe_symm.png'
                pdf_path = base_path + '_fe_symm.pdf'
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {png_path}")
                print(f"Saved: {pdf_path}")
            if show:
                plt.show()
    except Exception as e:
        print(f"Warning: could not create symmetrized FE figure: {e}")

    # === delta_F vs mass and save CSV ===
    try:
        delta_vals = []
        params = []
        for data, mval, mname in zip(all_data, mass_values, mass_names):
            if not data:
                params.append(mval)
                delta_vals.append(np.nan)
                continue
            xw, histw, few = get_wham_data_local(data)
            fe_arr = None
            if few is not None:
                fe_arr = np.asarray(few)
            elif "wham_plus_fe" in data:
                fe_arr = np.asarray(data.get("wham_plus_fe"))

            if fe_arr is None:
                params.append(mval)
                delta_vals.append(np.nan)
                continue

            valid = np.where(np.isfinite(fe_arr))[0]
            if valid.size > 0:
                delta = float(fe_arr[valid[-1]])
            else:
                delta = np.nan
            params.append(mval)
            delta_vals.append(delta)

        if any(np.isfinite(delta_vals)):
            figd, axd = plt.subplots(figsize=(6, 4))
            axd.plot(params, delta_vals, 'o-', color='tab:blue')
            axd.set_xlabel('mass')
            axd.set_ylabel('Delta F (final FE value) [kBT]')
            axd.set_title(f'{title} - DeltaF vs mass')
            axd.grid(True, alpha=0.3)
            if save_path:
                base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
                png_path = base_path + '_deltaF.png'
                pdf_path = base_path + '_deltaF.pdf'
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {png_path}")
                print(f"Saved: {pdf_path}")
            if save_path:
                csv_path = base_path + '_deltaF.csv'
                with open(csv_path, 'w') as fh:
                    fh.write('mass,delta_F\n')
                    for p, d in zip(params, delta_vals):
                        fh.write(f"{p},{d}\n")
                print(f"Saved: {csv_path}")
            if show:
                plt.show()
    except Exception as e:
        print(f"Warning: could not create deltaF figure: {e}")

    return fig


def compare_mass(
    parent_dir: Annotated[str, typer.Argument(help="Parent directory containing mass* folders with histograms/")] = ".",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    title: Annotated[str, typer.Option("-title", help="Plot title")] = "Mass Comparison",
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    log_scale: Annotated[bool, typer.Option("-log/--linear", help="Use log scale for histograms")] = False,
    cmap: Annotated[str, typer.Option("-cmap", help="Colormap name (viridis, plasma, coolwarm, etc.)")] = "plasma",
    plot_0minus: Annotated[bool, typer.Option("-0minus/--no-0minus", help="Include [0-] ensemble histogram")] = True,
    plot_0plus: Annotated[bool, typer.Option("-0plus/--no-0plus", help="Include [0+] (no WHAM) ensemble histogram")] = False,
):
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]

    save_path = save if save else None

    compare_mass_simulations(
        parent_dir=parent_dir,
        interfaces=intf_list,
        title=title,
        save_path=save_path,
        show=show,
        log_scale=log_scale,
        cmap=cmap,
        plot_0minus=plot_0minus,
        plot_0plus=plot_0plus,
    )


def plot_publication_free_energy(
    parent_dir: str,
    param_type: str = "gamma",
    interfaces: Optional[List[float]] = None,
    cmap: str = "plasma",
    save_path: Optional[str] = None,
    show: bool = True,
    use_scienceplots: bool = True,
    potential_type: str = "cosine",
    potential_height: float = 1.0,
    potential_center: float = 0.0,
    potential_width: float = 0.2,
    y_offset: float = 0.3,
    interface_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create publication-ready free energy comparison figure.
    
    Parameters
    ----------
    parent_dir : str
        Parent directory containing parameter* folders with histograms/
    param_type : str
        Type of parameter: 'gamma', 'subcycles', or 'mass'
    interfaces : list of float, optional
        Interface positions
    cmap : str
        Matplotlib colormap name (default: plasma)
    save_path : str, optional
        Path to save figure (PDF and PNG)
    show : bool
        Whether to display the figure
    use_scienceplots : bool
        Use scienceplots styling if available
    potential_type : str
        Type of potential overlay: 'line' or 'cosine'
    potential_height : float
        Height of potential bump (for cosine)
    potential_center : float
        Center of potential (default: 0.0)
    potential_width : float
        Width of cosine bump (default: 0.2)
    y_offset : float
        Vertical offset for zero line (shifts y=0 upward)
    interface_labels : list of str, optional
        Custom labels for interfaces (default: auto-generate)
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Find directories based on parameter type
    if param_type == "gamma":
        param_dirs = find_gamma_directories(parent_dir)
        label_prefix = r"$\gamma$"
    elif param_type == "subcycles":
        param_dirs = find_subcycles_directories(parent_dir)
        label_prefix = "FSR"
    elif param_type == "mass":
        param_dirs = find_mass_directories(parent_dir)
        label_prefix = "Mass"
    else:
        raise ValueError(f"Unknown param_type: {param_type}")
    
    if not param_dirs:
        raise ValueError(f"No {param_type}*/histograms directories found in {parent_dir}")
    
    dirs = [d[0] for d in param_dirs]
    param_values = [d[1] for d in param_dirs]
    param_names = [d[2] for d in param_dirs]
    
    # Create colormap avoiding too-light colors (use 0.2 to 0.95 range)
    colors = get_gamma_colormap(param_values, cmap, vmin=0.2, vmax=0.95)
    
    # Load all data
    all_data = []
    for d in dirs:
        try:
            data = load_histogram_data(d)
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load data from {d}: {e}")
            all_data.append({})
    
    # Set up publication style
    if use_scienceplots and SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['science', 'no-latex'])
    
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Small dashed line at the FE zero (shifted by y_offset)
    ax.axhline(y=y_offset, color='black', linestyle='-', linewidth=1.0, alpha=0.7, zorder=12)
    
    # Helper to get WHAM data
    def get_wham_data_local(data):
        if "wham_plus_x" in data:
            return data["wham_plus_x"], data.get("wham_plus_fe")
        return data.get("wham_x", np.array([])), data.get("wham_fe")
    
    # Plot free energies with subtle markers
    for data, pval, pname in zip(all_data, param_values, param_names):
        if not data:
            continue
        x, fe = get_wham_data_local(data)
        if x is None or fe is None or len(x) == 0:
            continue
        
        # Shift FE by y_offset
        fe_shifted = np.asarray(fe) + y_offset

        # Format label
        if param_type == "gamma":
            label = f"{label_prefix} = {pval:.0f} time$^{{-1}}$"
        elif param_type == "subcycles":
            label = f"{label_prefix} = {int(pval)}"
        else:
            label = f"{label_prefix} = {pval}"
        # Plot line without markers (publication style)
        ax.plot(x, fe_shifted, color=colors[pval], lw=1.8, label=label, zorder=10)
    
    # Plot potential
    # Potential overlay: either flat line or triangular/smooth bump between interfaces
    V0 = 0
    if potential_type == "line":
        ax.axhline(y_offset, color='black', lw=2.5, zorder=5, alpha=0.8)
        V0 = 0
    elif potential_type == "cosine":
        # Expect interfaces to contain at least three values: lambda_0, lambda_1, lambda_2
        V0 = 1
        xlim = ax.get_xlim()
        x_pot = np.linspace(xlim[0], xlim[1], 1000)
        V_pot = np.zeros_like(x_pot) + y_offset
        if interfaces and len(interfaces) >= 3:
            lam0, lam1, lam2 = interfaces[0], interfaces[1], interfaces[2]
            # rising edge between lam0->lam1, falling lam1->lam2
            rise_mask = (x_pot >= lam0) & (x_pot <= lam1)
            fall_mask = (x_pot >= lam1) & (x_pot <= lam2)
            # smooth rise using half-cosine
            if lam1 > lam0:
                xi = (x_pot[rise_mask] - lam0) / (lam1 - lam0)
                V_pot[rise_mask] = y_offset + 0.5 * (1 - np.cos(np.pi * xi))
            if lam2 > lam1:
                xi2 = (x_pot[fall_mask] - lam1) / (lam2 - lam1)
                V_pot[fall_mask] = y_offset + 0.5 * (1 + np.cos(np.pi * xi2))
        # plot without adding to legend
        ax.plot(x_pot, V_pot, color='black', lw=2.5, zorder=5, alpha=0.9)
    
    # Add interface lines with LaTeX labels
    if interfaces:
        if interface_labels is None:
            # Auto-generate labels: lambda_0, lambda_1, lambda_2, ...
            interface_labels = []
            for i, intf in enumerate(interfaces):
                interface_labels.append(f"$\\lambda_{{{i}}}$")
        
        ylim = ax.get_ylim()
        for intf, label in zip(interfaces, interface_labels):
            ax.axvline(intf, color='gray', alpha=0.4, ls='-', lw=1.2, zorder=3)
            ax.text(intf, ylim[0] + 3.1, label, ha='center', va='top', 
                   fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    
    # Labels and formatting
    ax.set_xlabel(r'Order parameter $\lambda$', fontsize=12)
    ax.set_ylabel(r'Conditional free energy $F(\lambda)$ [$k_{\mathrm{B}}T$]', fontsize=12)
    # Legend: replace 'sc' with 'FSR' already handled in label formatting
    ax.legend(fontsize=9, frameon=False, fancybox=False, loc='best', bbox_to_anchor=(0.03, 0.5, 0.4, 0.5))
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Set x-ticks at -0.1, 0.0, 0.1
    ax.set_xticks([-0.1, 0.0, 0.1])
    ax.minorticks_on()
    
    # Set y-ticks at integer Free Energy values (0.0, 1.0, ...), shifted by y_offset
    ymin, ymax = (-y_offset, max(fe_shifted) + 0.5)
    kmin = int(np.ceil(ymin - y_offset))
    kmax = int(np.floor(ymax - y_offset))
    ticks = np.arange(kmin, kmax + 1, 1.0) + y_offset
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{k:.1f}" for k in np.arange(kmin, kmax + 1)])

    plt.tight_layout()
    
    # Add annotation for potential value (V_0)
    V0_text = f"$V_0 = {int(V0)}$"
    ax.text(
        0.9,
        0.06,
        V0_text,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        color='black',
        fontsize=12,
        bbox=dict(facecolor='white', edgecolor='none', pad=0.4),
        zorder=20,
    )
    
    if save_path:
        base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
        png_path = base_path + '_publication.png'
        pdf_path = base_path + '_publication.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    
    if show:
        plt.show()
    
    # Reset style
    if use_scienceplots and SCIENCEPLOTS_AVAILABLE:
        plt.rcdefaults()
    
    return fig


def compare_gamma_publication(
    parent_dir: Annotated[str, typer.Argument(help="Parent directory containing gamma* folders")] = ".",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    cmap: Annotated[str, typer.Option("-cmap", help="Colormap name")] = "plasma",
    potential: Annotated[str, typer.Option("-potential", help="Potential type: 'line' or 'cosine'")] = "cosine",
    y_offset: Annotated[float, typer.Option("-yoffset", help="Vertical offset for zero line")] = 0.3,
):
    """
    Create publication-ready gamma comparison figure.
    """
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]
    
    save_path = save if save else None
    
    plot_publication_free_energy(
        parent_dir=parent_dir,
        param_type="gamma",
        interfaces=intf_list,
        cmap=cmap,
        save_path=save_path,
        show=show,
        potential_type=potential,
        y_offset=y_offset,
    )


def compare_subcycles_publication(
    parent_dir: Annotated[str, typer.Argument(help="Parent directory containing subcycle* folders")] = ".",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    cmap: Annotated[str, typer.Option("-cmap", help="Colormap name")] = "plasma",
    potential: Annotated[str, typer.Option("-potential", help="Potential type: 'line' or 'cosine'")] = "cosine",
    y_offset: Annotated[float, typer.Option("-yoffset", help="Vertical offset for zero line")] = 0.3,
):
    """
    Create publication-ready subcycles comparison figure.
    """
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]
    
    save_path = save if save else None
    
    plot_publication_free_energy(
        parent_dir=parent_dir,
        param_type="subcycles",
        interfaces=intf_list,
        cmap=cmap,
        save_path=save_path,
        show=show,
        potential_type=potential,
        y_offset=y_offset,
    )


def compare_mass_publication(
    parent_dir: Annotated[str, typer.Argument(help="Parent directory containing mass* folders")] = ".",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    cmap: Annotated[str, typer.Option("-cmap", help="Colormap name")] = "plasma",
    potential: Annotated[str, typer.Option("-potential", help="Potential type: 'line' or 'cosine'")] = "cosine",
    y_offset: Annotated[float, typer.Option("-yoffset", help="Vertical offset for zero line")] = 0.3,
):
    """
    Create publication-ready mass comparison figure.
    """
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]
    
    save_path = save if save else None
    
    plot_publication_free_energy(
        parent_dir=parent_dir,
        param_type="mass",
        interfaces=intf_list,
        cmap=cmap,
        save_path=save_path,
        show=show,
        potential_type=potential,
        y_offset=y_offset,
    )


# Create typer app with subcommands
app = typer.Typer(help="Plot histograms and free energies from infretis simulations")
app.command("plot")(plot_all)
app.command("compare")(compare)
app.command("compare_gamma")(compare_gamma)
app.command("compare_subcycles")(compare_subcycles)
app.command("compare_mass")(compare_mass)
app.command("compare_gamma_publication")(compare_gamma_publication)
app.command("compare_subcycles_publication")(compare_subcycles_publication)
app.command("compare_mass_publication")(compare_mass_publication)


def plot_publication_deltaF(
    csv_files: List[str],
    param_type: str = "gamma",
    cmap: str = "plasma",
    save_path: Optional[str] = None,
    show: bool = True,
    function_expr: Optional[str] = None,
    function_label: Optional[str] = None,
) -> plt.Figure:
    """
    Plot deltaF values from explicitly provided CSV files.

    This creates a minimal, publication-style plot with markers for sparse
    data points. No shifted y=0, no extra annotations.
    
    Parameters
    ----------
    csv_files : list of str
        List of CSV file paths to plot
    param_type : str
        Type of parameter for xlabel ('gamma', 'subcycles', 'mass')
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    function_expr : str, optional
        Python expression for function to plot (use 'x' as variable, e.g., '2*x + 1' or 'np.log(x)')
    function_label : str, optional
        Label for the function curve (default: 'Theory')
    """
    if not csv_files:
        raise ValueError("No CSV files provided")
    
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # Determine xlabel
    if param_type == "gamma":
        xlabel = r"Friction coefficient $\gamma$"
    elif param_type == "subcycles":
        xlabel = "Frame saving rate (FSR)"
    elif param_type == "mass":
        xlabel = "Mass"
    else:
        xlabel = param_type

    fig, ax = plt.subplots(figsize=(6, 4))

    # Create colormap for different CSV files (darker range)
    colors_list = plt.cm.plasma(np.linspace(0.15, 0.85, len(csv_files)))
    
    # Plot each CSV as a separate series
    for idx, csvp in enumerate(csv_files):
        if not os.path.exists(csvp):
            print(f"Warning: File not found: {csvp}")
            continue
            
        try:
            xcol, ycols, headers = load_csv(csvp)
        except Exception:
            try:
                data = np.genfromtxt(csvp, delimiter=',', skip_header=1)
                data = np.atleast_2d(data)
                if data.size == 0:
                    continue
                xcol = data[:, 0]
                ycols = data[:, 1:] if data.shape[1] > 1 else data[:, 0:1]
            except Exception as e:
                print(f"Warning: Could not load {csvp}: {e}")
                continue

    # Plot each CSV as a separate series
    for idx, csvp in enumerate(csv_files):
        if not os.path.exists(csvp):
            print(f"Warning: File not found: {csvp}")
            continue
            
        try:
            xcol, ycols, headers = load_csv(csvp)
        except Exception:
            try:
                data = np.genfromtxt(csvp, delimiter=',', skip_header=1)
                data = np.atleast_2d(data)
                if data.size == 0:
                    continue
                xcol = data[:, 0]
                ycols = data[:, 1:] if data.shape[1] > 1 else data[:, 0:1]
            except Exception as e:
                print(f"Warning: Could not load {csvp}: {e}")
                continue

        if xcol.size == 0:
            continue

        # Collect data from this CSV
        csv_params = []
        csv_deltaF = []
        for xi, row in zip(np.atleast_1d(xcol), np.atleast_2d(ycols)):
            try:
                param_val = float(xi)
                deltaF_val = float(row[0])
            except Exception:
                continue
            csv_params.append(param_val)
            csv_deltaF.append(deltaF_val)
        
        if csv_params:
            # Extract label from filename or use index
            label = os.path.basename(csvp).replace('.csv', '').replace('_', ' ')
            ax.plot(csv_params, csv_deltaF, 'o-', color=colors_list[idx], 
                   markersize=6, lw=1.2, label=label)
    
    if not ax.has_data():
        raise ValueError("No valid deltaF data found in provided CSV files")
    
    # Plot hardcoded function if provided
    if function_expr:
        xlim = ax.get_xlim()
        x_func = np.linspace(xlim[0], xlim[1], 500)
        try:
            # Evaluate the function expression
            # Available: x (variable), np (numpy)
            x = x_func  # noqa: F841
            y_func = eval(function_expr)
            func_label = function_label if function_label else 'Theory'
            ax.plot(x_func, y_func, 'k--', linewidth=2, label=func_label, zorder=1)
            ax.plot(x_func, -np.log(np.sqrt(2/x_func)/(4)), 'r--', linewidth=2, label="mass fit", zorder=1)
        except Exception as e:
            print(f"Warning: Could not evaluate function expression '{function_expr}': {e}")
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylabel(r'$\Delta F$ [$k_{\mathrm{B}}T$]', fontsize=12)
    ax.legend(fontsize=11, frameon=False, loc='best')
    ax.grid(True, alpha=0.25)
    ax.minorticks_on()
    
    # Set y-axis to start at 0
    ylim = ax.get_ylim()
    ax.set_ylim(0, ylim[1])

    plt.tight_layout()

    if save_path:
        base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
        png_path = base_path + '_deltaF_publication.png'
        pdf_path = base_path + '_deltaF_publication.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

    if show:
        plt.show()

    return fig


def compare_delta_publication(
    csv_files: Annotated[List[str], typer.Argument(help="CSV file paths to plot (space-separated)")],
    param_type: Annotated[str, typer.Option("-type", help="gamma|subcycles|mass")] = "gamma",
    cmap: Annotated[str, typer.Option("-cmap", help="Colormap name")] = "plasma",
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    function: Annotated[Optional[str], typer.Option("-function", help="Function to plot (use 'x' as variable, e.g., '2*x+1' or 'np.log(x)')")] = None,
    function_label: Annotated[Optional[str], typer.Option("-function-label", help="Label for function curve")] = None,
):
    """
    Plot deltaF from explicitly provided CSV files.
    
    Example:
        inft compare_delta_publication subcycle1/histograms/deltaF.csv subcycle2/histograms/deltaF.csv -type subcycles -save deltaF_pub
        inft compare_delta_publication data.csv -function "2*np.log(x)" -function-label "2*ln(x)"
    """
    save_path = save if save else None
    plot_publication_deltaF(
        csv_files=csv_files,
        param_type=param_type,
        cmap=cmap,
        save_path=save_path,
        show=show,
        function_expr=function,
        function_label=function_label,
    )


app.command("compare_delta_publication")(compare_delta_publication)


def plot_publication_comparison_grid(
    parent_dir_line_left: str,
    parent_dir_line_right: str,
    parent_dir_cosine_left: str,
    parent_dir_cosine_right: str,
    param_type_left: str = "gamma",
    param_type_right: str = "mass",
    interfaces: Optional[List[float]] = None,
    cmap_left: str = "plasma",
    cmap_right: str = "viridis",
    csv_pattern: str = "*deltaF.csv",
    csv_files_left: Optional[List[str]] = None,
    csv_files_right: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    y_offset: float = 0.3,
    columns: str = "both",
    log_scale_deltaF: bool = False,
    loglog_deltaF: bool = False,
) -> plt.Figure:
    """
    Create a 3x2 grid publication figure comparing two parameter types.
    
    Layout:
    - Row 1: Free energy with line potential (left: param_type_left, right: param_type_right)
    - Row 2: Free energy with cosine potential (left: param_type_left, right: param_type_right)
    - Row 3: DeltaF plot (left: param_type_left, right: param_type_right)
    
    Rows 1-2 share x-axis within each column.
    
    Args:
        parent_dir_line_left: Parent directory for left column line potential.
        parent_dir_line_right: Parent directory for right column line potential.
        parent_dir_cosine_left: Parent directory for left column cosine potential.
        parent_dir_cosine_right: Parent directory for right column cosine potential.
        csv_files_left: Optional list of CSV file paths for left column deltaF.
                       If provided, uses these instead of csv_pattern discovery.
        csv_files_right: Optional list of CSV file paths for right column deltaF.
                        If provided, uses these instead of csv_pattern discovery.
    """
    # Enable LaTeX rendering - must be done before creating the figure
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 15
    
    # Increase figure height to accommodate taller first row
    fig = plt.figure(figsize=(8, 11))
    
    # Create custom GridSpec with different spacing between row pairs
    # Rows 1-2 close together, row 3 further away
    # Row 1 is taller to accommodate extended y-axis
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 2, figure=fig, 
                          height_ratios=[1.4, 1, 1], 
                          hspace=0.0, wspace=0.0)
    
    # Create subplots manually with adjusted spacing
    axes = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            axes[i, j] = fig.add_subplot(gs[i, j])
    
    # Manually adjust positions to bring rows 1-2 closer
    for j in range(2):
        pos0 = axes[0, j].get_position()
        pos1 = axes[1, j].get_position()
        pos2 = axes[2, j].get_position()
        
        # Keep row 0 as is (already taller from height_ratios)
        new_pos0 = [pos0.x0, pos0.y0, pos0.width, pos0.height]
        # Position row 1 directly below row 0 - no gap
        new_pos1 = [pos1.x0, pos0.y0 - pos1.height, pos1.width, pos1.height]
        # Position row 2 directly below row 1 - then add small gap
        # Top of row 2 = Bottom of row 1 (new_pos1[1]) - gap (0.05)
        # Bottom of row 2 = Top - pos2.height
        new_pos2 = [pos2.x0, new_pos1[1] - 0.07 - pos2.height, pos2.width, pos2.height]
        
        axes[0, j].set_position(new_pos0)
        axes[1, j].set_position(new_pos1)
        axes[2, j].set_position(new_pos2)
    
    param_types = [param_type_left, param_type_right]
    
    for col_idx, param_type in enumerate(param_types):
        # Skip plotting this column if user requested a single column
        if columns == "left" and col_idx == 1:
            for r in range(3):
                axes[r, col_idx].set_visible(False)
            continue
        if columns == "right" and col_idx == 0:
            for r in range(3):
                axes[r, col_idx].set_visible(False)
            continue
        
        # Select parent directories based on column
        if col_idx == 0:  # left column
            parent_dir_line = parent_dir_line_left
            parent_dir_cosine = parent_dir_cosine_left
        else:  # right column
            parent_dir_line = parent_dir_line_right
            parent_dir_cosine = parent_dir_cosine_right
        
        # ===== Row 1: Line potential =====
        # Find directories for line potential
        if param_type == "gamma":
            param_dirs_line = find_gamma_directories(parent_dir_line)
            label_prefix = r"$\gamma$"
            xlabel_text = r"Friction coefficient $\gamma$"
        elif param_type == "subcycles":
            param_dirs_line = find_subcycles_directories(parent_dir_line)
            label_prefix = "FSR"
            xlabel_text = "Frame saving rate (FSR)"
        elif param_type == "mass":
            param_dirs_line = find_mass_directories(parent_dir_line)
            label_prefix = r"$m$"
            xlabel_text = r"Mass $m$"
        else:
            raise ValueError(f"Unknown param_type: {param_type}")
        
        if not param_dirs_line:
            print(f"Warning: No {param_type}* directories found in {parent_dir_line}")
        
        dirs_line = [d[0] for d in param_dirs_line] if param_dirs_line else []
        param_values_line = [d[1] for d in param_dirs_line] if param_dirs_line else []
        param_names_line = [d[2] for d in param_dirs_line] if param_dirs_line else []
        
        # choose colormap for this column
        cmap_name = cmap_left if col_idx == 0 else cmap_right
        colors_line = get_gamma_colormap(param_values_line, cmap_name, vmin=0.2, vmax=0.95) if param_values_line else {}
        
        # Load histogram data for line potential
        all_data_line = []
        for d in dirs_line:
            try:
                data = load_histogram_data(d)
                all_data_line.append(data)
            except Exception as e:
                all_data_line.append({})
        
        def get_wham_data_local(data):
            if "wham_plus_x" in data:
                return data["wham_plus_x"], data.get("wham_plus_fe")
            return data.get("wham_x", np.array([])), data.get("wham_fe")
        
        ax = axes[0, col_idx]
        ax.axhline(y=y_offset, color='black', linestyle='-', linewidth=1.0, alpha=0.7, zorder=12)
        
        for data, pval, pname in zip(all_data_line, param_values_line, param_names_line):
            if not data:
                continue
            x, fe = get_wham_data_local(data)
            if x is None or fe is None or len(x) == 0:
                continue
            
            fe_shifted = np.asarray(fe) + y_offset
            if param_type == "gamma":
                label = f"{label_prefix} = {pval:.0f}"
            elif param_type == "subcycles":
                label = f"{label_prefix} = {int(pval)}"
            else:
                label = f"{label_prefix} = {pval:.0f}" if pval >= 1 else (f"{label_prefix} = {pval:.2f}" if pval < 0.1 else f"{label_prefix} = {pval:.1f}")
            ax.plot(x, fe_shifted, color=colors_line[pval], lw=1.8, label=label, zorder=10)
        
        # Line potential overlay
        ax.axhline(y_offset, color='black', linestyle='--', lw=2.5, zorder=5, alpha=0.8)
        
        # Interface lines
        if interfaces:
            ylim = ax.get_ylim()
            for i, intf in enumerate(interfaces):
                ax.axvline(intf, color='gray', alpha=0.4, ls='-', lw=1.2, zorder=3)
                # Labels at top, centered on interface
                ax.text(intf-0.019 if (i == len(interfaces)-1) else intf+0.025 if (i==0) else intf, 5.9, f"$\\lambda_{{{i}}}$" if 0 < i < len(interfaces)-1 else (f"$\\lambda_{{{i}}}=\\lambda_A$" if i == 0 else f"$\\lambda_B=\\lambda_{{{i}}}$"), ha='center', va='top', 
                       fontsize=17, color='black', bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
        
        # ax.set_ylabel(r'$F(\lambda)$ [$k_{\mathrm{B}}T$]', fontsize=17)
        ax.set_ylabel(r'Energy ($k_{\mathrm{B}}T$)', fontsize=17)
        # Legend: reverse order for the first-panel in both columns
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=12.5, frameon=False, loc='center', bbox_to_anchor=(0.1, 0.13, 0.4, 0.5), labelspacing=.2)
        ax.set_xticks([-0.1, 0.0, 0.1])
        ax.set_xticklabels([])
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, bottom=True, left=True, right=True)
        
        # V_0 annotation
        ax.text(0.91, 0.053, r"$V_0 = 0$", transform=ax.transAxes, ha='right', va='bottom',
               color='black', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', pad=0.4), zorder=20)
        
        # ===== Add inset histogram panel for top-left (col_idx == 0, gamma only) =====
        if col_idx == 0 and param_type == "gamma":
            # Create inset axes in the empty space (upper right area)
            # Use absolute positioning for better control
            from matplotlib.patches import Rectangle
            
            # Position: [left, bottom, width, height] in axes coordinates
            pos = ax.get_position()
            fig_width, fig_height = fig.get_size_inches()
            
            # Calculate width relative to parent axes, but ensure physical squareness
            inset_width = pos.width * 0.4
            # h_rel = w_rel * (fig_width / fig_height) to make it physically square
            inset_height = inset_width * (fig_width / fig_height)*0.85
            
            ax_inset = fig.add_axes([pos.x0 + pos.width * 0.55,
                                     pos.y0 + pos.height * 0.72,
                                     inset_width,
                                     inset_height]) 
            
            # Set square aspect ratio (redundant but good for safety if limits change)
            # ax_inset.set_aspect('equal', adjustable='box')
            
            # Helper to get histogram data
            def get_wham_hist_local(data):
                if "wham_plus_x" in data:
                    return data["wham_plus_x"], data.get("wham_plus_hist")
                return data.get("wham_x", np.array([])), data.get("wham_hist")
            
            ax_inset.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=5)
            # Find gamma=1 and gamma=500 (or closest values)
            target_gammas = [1.0, 10, 500.0]
            hist_data = {}
            
            for target_gamma in target_gammas:
                # Find closest gamma value
                if param_values_line:
                    closest_idx = min(range(len(param_values_line)), 
                                    key=lambda i: abs(param_values_line[i] - target_gamma))
                    actual_gamma = param_values_line[closest_idx]
                    
                    # Only use if within reasonable range (e.g., within 20% or exact match)
                    if abs(actual_gamma - target_gamma) / target_gamma < 0.2 or actual_gamma == target_gamma:
                        data = all_data_line[closest_idx]
                        x_hist, hist = get_wham_hist_local(data)
                        
                        if x_hist is not None and hist is not None and len(x_hist) > 0:
                            # Find values at lambda = -0.1 and +0.1
                            idx_neg = np.argmin(np.abs(x_hist - (-0.1)))
                            idx_pos = np.argmin(np.abs(x_hist - 0.1))
                            
                            norm_factor = hist[idx_neg] + hist[idx_pos]
                            if norm_factor > 0:
                                hist_normalized = hist / norm_factor
                                hist_data[actual_gamma] = (x_hist, hist_normalized)
            
            # Plot normalized histograms in inset
            for gamma_val in sorted(hist_data.keys()):
                x_hist, hist_norm = hist_data[gamma_val]
                color = colors_line[gamma_val]
                ax_inset.plot(x_hist, hist_norm, color=color, lw=1.5, 
                            label=f"{label_prefix} = {gamma_val:.0f}")
            
            # Configure inset
            ax_inset.set_xlabel(r'$\lambda$', fontsize=12)
            ax_inset.set_ylabel(rf'$\varrho_\mathcal{{A}}$', fontsize=12)
            ax_inset.tick_params(labelsize=12, direction='in', which='both')
            # ax_inset.legend(fontsize=8, frameon=False, loc='upper left')
            # ax_inset.grid(True, alpha=0.2, linewidth=0.5)
            ax_inset.minorticks_on()
        
        # ===== Row 2: Cosine potential =====
        # Find directories for cosine potential
        if param_type == "gamma":
            param_dirs_cosine = find_gamma_directories(parent_dir_cosine)
        elif param_type == "subcycles":
            param_dirs_cosine = find_subcycles_directories(parent_dir_cosine)
        elif param_type == "mass":
            param_dirs_cosine = find_mass_directories(parent_dir_cosine)
        else:
            raise ValueError(f"Unknown param_type: {param_type}")
        
        if not param_dirs_cosine:
            print(f"Warning: No {param_type}* directories found in {parent_dir_cosine}")
        
        dirs_cosine = [d[0] for d in param_dirs_cosine] if param_dirs_cosine else []
        param_values_cosine = [d[1] for d in param_dirs_cosine] if param_dirs_cosine else []
        param_names_cosine = [d[2] for d in param_dirs_cosine] if param_dirs_cosine else []
        
        cmap_name = cmap_left if col_idx == 0 else cmap_right
        colors_cosine = get_gamma_colormap(param_values_cosine, cmap_name, vmin=0.2, vmax=0.95) if param_values_cosine else {}
        
        # Load histogram data for cosine potential
        all_data_cosine = []
        for d in dirs_cosine:
            try:
                data = load_histogram_data(d)
                all_data_cosine.append(data)
            except Exception as e:
                all_data_cosine.append({})
        
        ax = axes[1, col_idx]
        ax.axhline(y=y_offset, color='black', linestyle='-', linewidth=1.0, alpha=0.7, zorder=12)
        
        for data, pval, pname in zip(all_data_cosine, param_values_cosine, param_names_cosine):
            if not data:
                continue
            x, fe = get_wham_data_local(data)
            if x is None or fe is None or len(x) == 0:
                continue
            
            fe_shifted = np.asarray(fe) + y_offset
            if param_type == "gamma":
                label = f"{label_prefix} = {pval:.0f}"
            elif param_type == "subcycles":
                label = f"{label_prefix} = {int(pval)}"
            else:
                label = f"{label_prefix} = {pval:.1f}"
            ax.plot(x, fe_shifted, color=colors_cosine[pval], lw=1.8, label=label, zorder=10)
        
        # Cosine potential overlay
        xlim = ax.get_xlim()
        x_pot = np.linspace(xlim[0], xlim[1], 1000)
        V_pot = np.zeros_like(x_pot) + y_offset
        if interfaces and len(interfaces) >= 3:
            lam0, lam1, lam2 = interfaces[0], interfaces[1], interfaces[2]
            rise_mask = (x_pot >= lam0) & (x_pot <= lam1)
            fall_mask = (x_pot >= lam1) & (x_pot <= lam2)
            if lam1 > lam0:
                xi = (x_pot[rise_mask] - lam0) / (lam1 - lam0)
                V_pot[rise_mask] = y_offset + 0.5 * (1 - np.cos(np.pi * xi))
            if lam2 > lam1:
                xi2 = (x_pot[fall_mask] - lam1) / (lam2 - lam1)
                V_pot[fall_mask] = y_offset + 0.5 * (1 + np.cos(np.pi * xi2))
        ax.plot(x_pot, V_pot, color='black', linestyle='--', lw=2.5, zorder=5, alpha=0.9)
        
        # Interface lines
        if interfaces:
            ylim = ax.get_ylim()
            for i, intf in enumerate(interfaces):
                ax.axvline(intf, color='gray', alpha=0.4, ls='-', lw=1.2, zorder=3)
                # Labels at top, centered on interface
                # ax.text(intf, ylim[1] - 0.8, f"$\\lambda_{{{i}}}$" if 0 < i < len(interfaces)-1 else (f"$\\lambda_{{{i}}}=\\lambda_A$" if i == 0 else f"$\\lambda_B=\\lambda_{{{i}}}$"), ha='center', va='top', 
                #        fontsize=17, color='black', bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
        
        ax.set_xlabel(r'$\lambda=x$', fontsize=17)
        ax.set_ylabel(r'Energy ($k_{\mathrm{B}}T$)', fontsize=17)
        # ax.legend(fontsize=9, frameon=False, loc='best', bbox_to_anchor=(0.03, 0.5, 0.4, 0.5))
        ax.set_xticks([-0.1, 0.0, 0.1])
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, bottom=True, left=True, right=True)
        
        # V_0 annotation
        ax.text(0.9, 0.18, r"$V_0 = 1$", transform=ax.transAxes, ha='right', va='bottom',
               color='black', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', pad=0.4), zorder=20)
        
        # Ensure rows 1 and 2 share identical x- and y-limits for this column
        try:
            ax_top = axes[0, col_idx]
            ax_mid = axes[1, col_idx]
            x0 = ax_top.get_xlim()
            x1 = ax_mid.get_xlim()
            new_xlim = (min(x0[0], x1[0]), max(x0[1], x1[1]))
            y0 = ax_top.get_ylim()
            y1 = ax_mid.get_ylim()
            new_ylim = (min(y0[0], y1[0]), max(y0[1], y1[1])+0.1)
            ax_top.set_xlim(new_xlim)
            ax_mid.set_xlim(new_xlim)
            ax_top.set_ylim(new_ylim)
            ax_mid.set_ylim(new_ylim)
        except Exception:
            pass
        
        # ===== Row 3: DeltaF plot =====
        ax = axes[2, col_idx]
        
        deltaF_params = []
        deltaF_vals = []
        
        # Use explicit CSV files if provided, otherwise use pattern-based discovery
        csv_files_to_use = None
        if col_idx == 0 and csv_files_left:
            csv_files_to_use = csv_files_left
        elif col_idx == 1 and csv_files_right:
            csv_files_to_use = csv_files_right
        
        labels = [r"$V_0=0$", r"$V_0=1$"]
        markers = ['X', 'o', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        if csv_files_to_use:
            # Explicit CSV files: plot each with different markers and color gradient
            # Get the colormap and parameter values used in rows 1-2
            cmap_name = cmap_left if col_idx == 0 else cmap_right
            cmap_obj = plt.cm.get_cmap(cmap_name)
            
            # Get parameter values from line potential (used in rows 1-2)
            if col_idx == 0:
                param_vals_for_colors = param_values_line if param_values_line else []
            else:
                param_vals_for_colors = param_values_line if param_values_line else []
            
            # record which marker and marker sizes were used per CSV so we can
            # create matching legend proxies later (avoid querying artist types)
            legend_markers = []
            legend_marker_sizes = []
            for idx, csvp in enumerate(csv_files_to_use):
                try:
                    xcol, ycols, headers = load_csv(csvp)
                except Exception:
                    try:
                        data = np.genfromtxt(csvp, delimiter=',', skip_header=1)
                        data = np.atleast_2d(data)
                        xcol = data[:, 0]
                        ycols = data[:, 1:] if data.shape[1] > 1 else data[:, 0:1]
                    except Exception:
                        continue
                
                if np.size(xcol) == 0:
                    continue
                
                # Extract all rows from this CSV
                params = []
                deltas = []
                for xi, row in zip(np.atleast_1d(xcol), np.atleast_2d(ycols)):
                    try:
                        param_val = float(xi)
                        deltaF_val = float(row[0])
                        params.append(param_val)
                        deltas.append(deltaF_val)
                    except Exception:
                        continue
                
                if params:
                    # Create label from filename
                    from pathlib import Path
                    label = labels[idx] if idx < len(labels) else Path(csvp).stem.replace('_', ' ')
                    marker = markers[idx % len(markers)]
                    
                    # Plot with smooth color gradient matching row 1-2 evolution
                    if len(params) > 1 and param_vals_for_colors:
                        # Color based on data point index (0 to n-1), not parameter value
                        n_points = len(params)
                        
                        # Normalize indices to colormap range [0.2, 0.95] matching row 1-2
                        def norm_index(i):
                            if n_points > 1:
                                t = i / (n_points - 1.0)
                                return 0.2 + 0.75 * t
                            return 0.5
                        
                        # per-point colors based on index
                        point_colors = [cmap_obj(norm_index(i)) for i in range(n_points)]
                        
                        # Create smooth gradient by interpolating between points
                        from matplotlib.collections import LineCollection
                        # Interpolate to create many segments for smooth color transition
                        n_interp = 50  # segments per original segment
                        all_segs = []
                        all_colors = []
                        
                        for i in range(n_points - 1):
                            # Interpolate between point i and i+1
                            x0, y0 = params[i], deltas[i]
                            x1, y1 = params[i+1], deltas[i+1]
                            
                            for j in range(n_interp):
                                t_seg = j / n_interp
                                t_next = (j + 1) / n_interp
                                
                                # Use geometric interpolation for log scale axes
                                if (log_scale_deltaF or loglog_deltaF) and x0 > 0 and x1 > 0:
                                    x_seg0 = x0 * ((x1 / x0) ** t_seg)
                                    x_seg1 = x0 * ((x1 / x0) ** t_next)
                                else:
                                    x_seg0 = x0 + t_seg * (x1 - x0)
                                    x_seg1 = x0 + t_next * (x1 - x0)
                                
                                # Use geometric interpolation for y-axis in log-log mode
                                if loglog_deltaF and y0 > 0 and y1 > 0:
                                    y_seg0 = y0 * ((y1 / y0) ** t_seg)
                                    y_seg1 = y0 * ((y1 / y0) ** t_next)
                                else:
                                    y_seg0 = y0 + t_seg * (y1 - y0)
                                    y_seg1 = y0 + t_next * (y1 - y0)
                                
                                all_segs.append([(x_seg0, y_seg0), (x_seg1, y_seg1)])
                                
                                # Color based on overall position in data
                                overall_t = (i + t_seg + (t_next - t_seg) / 2.0) / (n_points - 1)
                                all_colors.append(cmap_obj(0.2 + 0.75 * overall_t))
                        
                        if all_segs:
                            lc = LineCollection(all_segs, colors=all_colors, linewidths=3, zorder=5)
                            ax.add_collection(lc)
                        
                        # Scatter points with index-based colors and add legend entry
                        pts = ax.scatter(params, deltas, marker=marker, c=point_colors, s=85, zorder=10, edgecolors='white', linewidths=0.5)
                        pts.set_label(label)
                        # record marker and approximate display size for legend proxy
                        legend_markers.append(marker)
                        try:
                            legend_marker_sizes.append(int(max(6, np.sqrt(85))))
                        except Exception:
                            legend_marker_sizes.append(8)
                    else:
                        # Fallback to single color
                           color = cmap_obj(0.5)
                           ax.plot(params, deltas, marker=marker, color=color, 
                               markersize=20, lw=0, label=label, linestyle='none')
                           legend_markers.append(marker)
                           legend_marker_sizes.append(10)
            
            if csv_files_to_use:
                # Reverse legend order and ensure legend markers are drawn with black edges
                handles, labels_leg = ax.get_legend_handles_labels()
                rev_handles = handles[::-1]
                rev_labels = labels_leg[::-1]
                # Create black proxy artists for the legend so only legend symbols are black
                import matplotlib.lines as mlines
                proxy_handles = []
                # Use recorded markers/sizes in the same (reversed) order as labels
                rev_markers = legend_markers[::-1]
                rev_marker_sizes = legend_marker_sizes[::-1]
                for i, lab in enumerate(rev_labels):
                    marker = rev_markers[i] if i < len(rev_markers) else 'o'
                    msize = rev_marker_sizes[i] if i < len(rev_marker_sizes) else 8
                    proxy = mlines.Line2D([], [], color='black', marker=marker, linestyle='None',
                                          markerfacecolor='black', markeredgecolor='black', markersize=msize)
                    proxy_handles.append(proxy)
                legend = ax.legend(proxy_handles, rev_labels, fontsize=16, frameon=False, loc='lower right')
        else:
            # Pattern-based discovery: combine all into single series
            # Use line potential directories for pattern-based discovery
            for dpath, pval in zip(dirs_line, param_values_line):
                matches = glob.glob(os.path.join(dpath, csv_pattern))
                if not matches:
                    try:
                        from pathlib import Path
                        matches = [str(p) for p in Path(dpath).rglob(csv_pattern)]
                    except Exception:
                        matches = []
                
                if not matches:
                    continue
                
                csvp = matches[0]
                try:
                    xcol, ycols, headers = load_csv(csvp)
                except Exception:
                    try:
                        data = np.genfromtxt(csvp, delimiter=',', skip_header=1)
                        data = np.atleast_2d(data)
                        xcol = data[:, 0]
                        ycols = data[:, 1:] if data.shape[1] > 1 else data[:, 0:1]
                    except Exception:
                        continue

                if np.size(xcol) == 0:
                    continue

                # collect all rows
                for xi, row in zip(np.atleast_1d(xcol), np.atleast_2d(ycols)):
                    try:
                        param_val = float(xi)
                        deltaF_val = float(row[0])
                    except Exception:
                        continue
                    deltaF_params.append(param_val)
                    deltaF_vals.append(deltaF_val)
            
            if deltaF_params:
                # use column colormap for combined series
                cmap_name = cmap_left if col_idx == 0 else cmap_right
                cmap_obj = plt.cm.get_cmap(cmap_name)
                color_use = cmap_obj(0.5)
                ax.plot(deltaF_params, deltaF_vals, 'o-', color=color_use, markersize=20, lw=1.2)

        
        ax.set_xlabel(xlabel_text, fontsize=17)
        
        # Apply log scale to axes if requested
        if loglog_deltaF:
            ax.set_xscale('log')
            ax.set_yscale('log')
        elif log_scale_deltaF:
            ax.set_xscale('log')
        
        ax.set_ylabel(r'$\Delta F_\mathcal{A}$ ($k_{\mathrm{B}}T$)', fontsize=17)
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, bottom=True, left=True, right=True)
        
        # Set y-axis to start at 0 (only for linear y-scale)
        if not loglog_deltaF:
            ylim = ax.get_ylim()
            ax.set_ylim(bottom=0)
    
    # Don't use tight_layout as we manually positioned axes
    # plt.tight_layout()
    # Align y-limits between left and right columns and put right-column y-axis on the
    # outer (right) side so the two columns share a common vertical axis visually.
    try:
        for r in range(3):
            axL = axes[r, 0]
            axR = axes[r, 1]
            # sync y-limits
            yL = axL.get_ylim()
            yR = axR.get_ylim()
            new_ylim = (min(yL[0], yR[0]), max(yL[1], yR[1]))
            
            # Extend top y-limit for first row by 0.8
            if r == 0:
                new_ylim = (new_ylim[0], new_ylim[1] + 0.8)
            
            axL.set_ylim(new_ylim)
            axR.set_ylim(new_ylim)
            # If this is a free-energy row (top two rows), set integer yticks shifted by y_offset
            try:
                if r in (0, 1):
                    ymin, ymax = new_ylim
                    kmin = int(np.ceil(ymin - y_offset))
                    kmax = int(np.floor(ymax - y_offset))
                    ticks = np.arange(kmin, kmax + 1, 1.0) + y_offset
                    axL.set_yticks(ticks)
                    axL.set_yticklabels([f"{k:.1f}" for k in np.arange(kmin, kmax + 1)])
                    # apply same ticks to right axis but without labels
                    axR.set_yticks(ticks)
                    axR.set_yticklabels(["" for _ in ticks])
            except Exception:
                pass
            # disable y-axis label and tick labels on the right column
            try:
                axR.set_ylabel("")
                axR.set_yticklabels([])
                axR.tick_params(axis='y', which='both', labelleft=False, labelright=False)
            except Exception:
                pass
    except Exception:
        pass

    if save_path:
        base_path = (save_path.rsplit('.', 1)[0] if '.' in save_path else save_path)
        png_path = base_path + '_grid.png'
        pdf_path = base_path + '_grid.pdf'
        svg_path = base_path + '_grid.svg'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
        print(f"Saved: {svg_path}")
    
    if show:
        plt.show()
    
    return fig


def compare_grid_publication(
    parent_dir_line_left: Annotated[str, typer.Argument(help="Parent directory for left column line potential")],
    parent_dir_line_right: Annotated[str, typer.Argument(help="Parent directory for right column line potential")],
    parent_dir_cosine_left: Annotated[str, typer.Argument(help="Parent directory for left column cosine potential")],
    parent_dir_cosine_right: Annotated[str, typer.Argument(help="Parent directory for right column cosine potential")],
    left: Annotated[str, typer.Option("-left", help="Left column param type (gamma/subcycles/mass)")] = "gamma",
    right: Annotated[str, typer.Option("-right", help="Right column param type (gamma/subcycles/mass)")] = "mass",
    interfaces: Annotated[Optional[str], typer.Option("-interfaces", help="Comma-separated interface values")] = None,
    csv_pattern: Annotated[str, typer.Option("-pattern", help="Glob pattern for deltaF CSVs (used if no explicit files)")] = "*deltaF.csv",
    csv_left: Annotated[Optional[str], typer.Option("-csv-left", help="Comma-separated CSV files for left column deltaF")] = None,
    csv_right: Annotated[Optional[str], typer.Option("-csv-right", help="Comma-separated CSV files for right column deltaF")] = None,
    cmap_left: Annotated[str, typer.Option("-cmap-left", help="Colormap for left column")] = "plasma",
    cmap_right: Annotated[str, typer.Option("-cmap-right", help="Colormap for right column")] = "viridis",
    save: Annotated[str, typer.Option("-save", help="Save figure to this path")] = "",
    show: Annotated[bool, typer.Option("-show/--no-show", help="Display figure")] = True,
    y_offset: Annotated[float, typer.Option("-yoffset", help="Vertical offset for zero line")] = 0.3,
    columns: Annotated[str, typer.Option("-columns", help="Which columns to plot: left|right|both")] = "both",
    log_deltaF: Annotated[bool, typer.Option("-log-deltaF/--linear-deltaF", help="Use log scale for deltaF x-axis")] = False,
    loglog_deltaF: Annotated[bool, typer.Option("-loglog-deltaF/--no-loglog-deltaF", help="Use log-log scale for deltaF plot")] = False,
):
    """
    Create 3x2 grid comparison figure with line/cosine/deltaF for two parameter types.
    
    Examples:
        inft compare_grid_publication ./gamma_line ./mass_line ./gamma_cosine ./mass_cosine -left gamma -right mass -csv-left file1.csv,file2.csv -csv-right file3.csv,file4.csv
    """
    intf_list = None
    if interfaces:
        intf_list = [float(x.strip()) for x in interfaces.split(",")]
    
    csv_files_left = None
    if csv_left:
        csv_files_left = [f.strip() for f in csv_left.split(",")]
    
    csv_files_right = None
    if csv_right:
        csv_files_right = [f.strip() for f in csv_right.split(",")]
    
    save_path = save if save else None
    
    plot_publication_comparison_grid(
        parent_dir_line_left=parent_dir_line_left,
        parent_dir_line_right=parent_dir_line_right,
        parent_dir_cosine_left=parent_dir_cosine_left,
        parent_dir_cosine_right=parent_dir_cosine_right,
        param_type_left=left,
        param_type_right=right,
        interfaces=intf_list,
        cmap_left=cmap_left,
        cmap_right=cmap_right,
        csv_pattern=csv_pattern,
        csv_files_left=csv_files_left,
        csv_files_right=csv_files_right,
        save_path=save_path,
        show=show,
        y_offset=y_offset,
        columns=columns,
        log_scale_deltaF=log_deltaF,
        loglog_deltaF=loglog_deltaF,
    )


app.command("compare_grid_publication")(compare_grid_publication)


if __name__ == "__main__":
    app()
