import os

import matplotlib.pyplot as plt

STYLE_PATH = os.path.join(os.path.dirname(__file__), "chemevo.mplstyle")


def use():
    """Apply the shared chemevo plotting rcParams (labels, ticks, fonts, ...)."""
    plt.style.use(STYLE_PATH)
