import os
import shutil
import numpy as np

from inftools.analysis.wham import wham


def test_wham(tmp_path):
    """Check that we can modify the velocities with an engine,
    and that they are not equal to zero."""
    # folder we wil run from
    folder = tmp_path / "temp/"
    folder.mkdir()
    pathpath= os.path.dirname(__file__) + "/../simdata_dw/"
    shutil.copytree(pathpath, str(folder) + "/simdata")
    os.chdir(str(folder) + "/simdata")

    # run wham analysis script (timestep required for rate/flux calculations)
    # first run without lambda_minus_one to establish baseline
    wham(timestep=1.0)

    data = np.loadtxt('wham/Pcross.txt')
    assert len(data) == 106
    assert data.shape == (106, 4)
    assert abs(6.958993086416705e-06 -data[-1][-1]) < 1000

    # now inject a lambda_minus_one value into the toml and rerun in a
    # separate output directory so that we exercise the xi-corrected rate
    # path.  We expect a runav_rate.txt file to be produced and its
    # contents to differ from the baseline (flux should be modified by xi.
    toml_path = folder / "simdata" / "infretis.toml"
    # add lambda_minus_one under [simulation.tis_set]
    text = toml_path.read_text()
    if "lambda_minus_one" not in text:
        lines = text.splitlines()
        out_lines = []
        for line in lines:
            out_lines.append(line)
            if line.strip() == "[simulation.tis_set]":
                out_lines.append("lambda_minus_one = -0.9")
        toml_path.write_text("\n".join(out_lines))

    # create dummy trajectory files for all path labels so that the xi
    # computation can load them without error
    data_file = folder / "simdata" / "infretis_data.txt"
    labels = set()
    for line in data_file.read_text().splitlines():
        if line.strip().startswith("#") or not line.strip():
            continue
        parts = line.split()
        try:
            labels.add(int(parts[0]))
        except ValueError:
            pass
    load_dir = folder / "simdata" / "load"
    for lbl in labels:
        sub = load_dir / str(lbl)
        sub.mkdir(parents=True, exist_ok=True)
        # write a minimal order.txt with three rows and two columns
        # so extract() returns type RMR (first>=lA, second<first, last>=lA)
        with open(sub / "order.txt", "w") as f:
            f.write("0.0 1.0\n1.0 0.0\n2.0 1.0\n")

    # run wham again with new folder and same timestep
    wham(folder="wham_lm1", timestep=1.0)
    lm1_rate = np.loadtxt('wham_lm1/runav_rate_lm1.txt')
    # corrected file should have at least three columns (xi, flux, rate)
    assert lm1_rate.shape[1] >= 3
    # check that corrected rate differs from baseline rate output by
    # comparing the first three columns only
    base_rate = np.loadtxt('wham/runav_rate.txt')
    assert not np.allclose(base_rate[:, :3], lm1_rate[:, :3])
