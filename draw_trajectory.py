import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{polski}",
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[polish]{babel}",
            ]
        ),
    }
)
import matplotlib.pyplot as plt


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


df = pd.read_csv(f"data/kitti_superpoint_supergluematch.txt")

print(df)

fig, ax = plt.subplots(1, 1, figsize=set_size(455))

# plot x and z
ax.plot(df["gtx"], df["gtz"], label=r"\textit{Ground truth}")
ax.plot(df["tx"], df["tz"], label="Przewidywana ścieżka")
ax.scatter(0, 0, label="Początek sekwencji", c="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()
ax.legend()

plt.savefig("data/trajektoria.png", dpi=300)
plt.savefig("data/trajektoria.pgf")
