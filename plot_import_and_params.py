import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors, cm, rcParams
params = {
    "text.usetex": True,
    "font.family": "CMU serif",
    "font.serif": ["Computer Modern Serif"],
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,  # a little smaller
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5
}
rcParams.update(params)
plt.rc("figure", titlesize=10)
plt.rc("text.latex", preamble=r"\usepackage{siunitx} \usepackage[T1]{fontenc} \usepackage{xcolor}")

"""This file is included into all plotting .py-files. It changes the font from the not-to-great-looking DejaVu Sans 
to CMU serif. In addition, some standard font settings are changed."""
