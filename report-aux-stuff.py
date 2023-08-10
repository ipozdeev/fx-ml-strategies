import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from config import *
from mpl_config import *

from src.datafeed.downstream import get_fx_data


def gen_tsh_example():
    """"""
    dt = "2023-05-05"
    ts_ex = (get_fx_data()\
        .get("term_structure_history")\
        .drop(["2y", "1w"], axis=1, level=1))\
        .loc[:dt, "aud"]\
        .tail(6) \
        .pipe(np.log) \
        .rename(index=lambda _x: _x.date())

    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.heatmap(ts_ex, annot=True, fmt=".4f", linewidths=0.3, cmap="coolwarm",
                cbar=False)
    ax.set_xlabel("tenor")
    ax.set_title(f"example TSH: AUDUSD, as of {dt}")
    fig.tight_layout()
    fig.savefig("reports/figures/tsh-example.png", dpi=200)


def gen_carry_conv():
    """"""
    dt = "2023-05-05"
    dt = "2023-05-05"
    ts_ex = (get_fx_data() \
        .get("term_structure_history") \
        .drop(["2y", "1w"], axis=1, level=1)) \
        .loc[:dt, "aud"] \
        .tail(6) \
        .rename(index=lambda _x: _x.date())
    carry_conv = ts_ex * 0.0
    carry_conv.iloc[-1].loc["1m"] = -1.0
    carry_conv.iloc[-1].loc["spot"] = +1.0

    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.heatmap(carry_conv, annot=True, fmt="+.1f", linewidths=0.3, cmap="coolwarm",
                cbar=False)
    ax.set_xlabel("tenor")
    ax.set_title(f"extracting the 1-month carry")
    fig.tight_layout()
    fig.savefig("reports/figures/carry-convolution.png", dpi=200)


if __name__ == '__main__':
    gen_carry_conv()
    gen_tsh_example()
