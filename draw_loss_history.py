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


MODEL = "img_stride_nodeindex"

df = pd.read_csv(f"models_{MODEL}/losses.txt")
df2 = pd.read_csv(f"models_{MODEL}_round2/losses.txt")
print(df)
print(df2)

df = pd.concat([df, df2])
df = df.reset_index(drop=True)
df["epoch"] = df.index + 1

fig, ax = plt.subplots(1, 1, figsize=set_size(455))
fig.subplots_adjust(wspace=0.4)

ax.plot(df["epoch"], df["train_loss"], label="Trening")
ax.plot(df["epoch"], df["eval_loss"], label="Walidacja")

# draw moving average
window_size = 5
eval_loss = df["eval_loss"].rolling(10).mean()
# lighter, dashed line
ax.plot(df["epoch"], eval_loss, label="Walidacja (średnia krocząca)", linestyle="--")

ax.set_xlabel("Epoka")
ax.set_ylabel("Wartość funkcji straty")

ax.legend()

fig.savefig(f"data/losses_{MODEL}.pgf")
fig.savefig(f"data/losses_{MODEL}.png")
