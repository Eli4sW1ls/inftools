"""The functions to be used to run infretis via the terminal."""
import pathlib
import sys

import typer

from inftools.misc.binhelper import get_mapper
from inftools.analysis.wham import wham
from inftools.analysis.compute_histograms import compute_all_histograms
from inftools.analysis.plot_histograms import plot_all as plot_histograms
from inftools.analysis.plot_histograms import compare as compare_histograms
from inftools.analysis.plot_histograms import compare_gamma
from inftools.analysis.plot_histograms import compare_subcycles
from inftools.analysis.plot_histograms import compare_mass
from inftools.analysis.plot_histograms import compare_gamma_publication
from inftools.analysis.plot_histograms import compare_subcycles_publication
from inftools.analysis.plot_histograms import compare_mass_publication
from inftools.analysis.plot_histograms import compare_delta_publication
from inftools.analysis.plot_histograms import compare_grid_publication
from inftools.analysis.rec_error_staple import error_analysis_staple

# define constants
MOD_PATH = str(pathlib.Path(__file__).parent.resolve())
FOLDERS = ["exercises", "tistools", "xyz", "report"]
MAPPER = get_mapper(FOLDERS, MOD_PATH)

# add individual functions
MAPPER["wham"] = wham
MAPPER["compute-histograms"] = compute_all_histograms
MAPPER["plot-histograms"] = plot_histograms
MAPPER["compare-histograms"] = compare_histograms
MAPPER["compare_gamma"] = compare_gamma
MAPPER["compare_subcycles"] = compare_subcycles
MAPPER["compare_mass"] = compare_mass
MAPPER["compare_gamma_publication"] = compare_gamma_publication
MAPPER["compare_subcycles_publication"] = compare_subcycles_publication
MAPPER["compare_mass_publication"] = compare_mass_publication
MAPPER["compare_delta_publication"] = compare_delta_publication
MAPPER["compare_grid_publication"] = compare_grid_publication
MAPPER["error_analysis_staple"] = error_analysis_staple

app = typer.Typer(
    no_args_is_help=True,
    help="inftools CLI",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# decorating imported mapper functions
for func in MAPPER.values():
    app.command()(func)

# NOTE: when defining new functionality
# put the import statements in the function defenition
# as to avoid importing loads of libraries, which slows
# down the `inft` call from the command line
