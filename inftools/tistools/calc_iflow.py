from typing import Annotated

import typer

# Disable automatic underscore -> hyphen in CLI names
# typer.main.get_command_name = lambda name: name

def calc_iflow(
    plot: Annotated[str, typer.Option("-plot", help="Plot the flow for those paths, string of spaced idxes")]="",
    log: Annotated[str, typer.Option("-log", help="The .log file to read path numbers")] = "sim.log",
    ):
    """
    Calculates and plots the flow of individual replica across ensembles.

    Returns a flow_map dictionary.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # import scienceplots
    # plt.style.use('science')

    from inftools.misc.misc import read_log
    from inftools.tistools.max_op import COLS

    for idx0, rep in enumerate([int(i) for i in plot.split(" ")]):

        ens, pns, follow_size, shootings, pns_uniq, ens_uniq, shootings2, imcsteps = read_log(log, rep)
        print("ens:", " ".join(pns_uniq))
        print("pns:", " ".join(ens_uniq))

        plt.plot(shootings2, [int(i) for i in ens_uniq], color=COLS[idx0%len(COLS)], zorder=10)
        plt.scatter(shootings2, [int(i) for i in ens_uniq], color=COLS[idx0%len(COLS)], zorder=10, s=15, edgecolor="k")
        if idx0 == 0:
            accum = np.cumsum([0] + imcsteps)
            for idx, (istep, fs) in enumerate(zip(imcsteps, follow_size)):
                plt.plot([accum[idx], accum[idx+1]], [fs[0]]*2, color="k", ls="--")
                if idx < len(imcsteps)-1:
                    dfs = follow_size[idx+1][0]-follow_size[idx][0]
                    plt.plot([accum[idx+1]]*2, [0, follow_size[idx][0] + dfs], color="k", ls="--", alpha=0.1)
                    if dfs == 0:
                        continue
                    plt.plot([accum[idx+1]]*2, [follow_size[idx][0], follow_size[idx][0] + dfs], color="k", ls="--")


    plt.xlabel("MC Moves")
    plt.ylabel("Ensemble")
    plt.ylim([0, None])
    plt.savefig("iflow.png")
    plt.show()
