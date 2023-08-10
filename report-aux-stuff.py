import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import torch

from config import *
from mpl_config import *

from src.datafeed.downstream import get_fx_data
from src.models import normalized_softmax
from src.finance import get_long_short_legs

palette = sns.color_palette("colorblind")
_blue = palette[0]
_red = palette[1]


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
    sns.heatmap(ts_ex, annot=True, fmt=".4f", linewidths=0.3, cmap="Greys",
                cbar=False)
    ax.set_xlabel("tenor")
    ax.set_title(f"example TSH: AUDUSD, as of {dt}")
    fig.tight_layout()
    fig.savefig("reports/figures/tsh-example.png", dpi=200)


def gen_carry_conv():
    """"""
    dt = "2023-05-05"
    dt = "2023-05-05"
    ts_ex = get_fx_data() \
        .get("term_structure_history") \
        .drop(["2y", "1w"], axis=1, level=1) \
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


def carry_from_conv():
    """"""
    data = get_fx_data()

    # carry
    fd = data["fd_annualized"].xs("1m", axis=1, level=1)
    rx = data["excess_returns"]

    # to weights
    w = normalized_softmax(
        torch.from_numpy(fd.values)
    )
    w = pd.DataFrame(w, index=fd.index, columns=fd.columns)

    rx_carry = rx.mul(w).sum(axis=1, min_count=1)

    # 'normal' carry
    rx_carry_normal = get_long_short_legs(fd)\
        .mul(rx).sum(axis=1, min_count=1)

    fig, ax = plt.subplots(figsize=(5, 2.5))

    for _p in range(22):
        rx_carry.iloc[_p::22].dropna().add(1).cumprod()\
            .plot(alpha=0.4, color=_red, ax=ax)
        rx_carry_normal.iloc[_p::22].dropna().add(1).cumprod()\
            .plot(alpha=0.4, color=_blue, ax=ax)
        
    # Create custom legend handles
    legend_elements = [
        Line2D([], [], color=_red, label='softmax-based'),
        Line2D([], [], color=_blue, label='rank-based')
    ]

    # Add the legend to the plot using the custom handles
    ax.legend(handles=legend_elements, loc='best')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_ylabel("cumulative return")
    ax.set_title("performance of carry strategies (monthly returns)")
                
    avg_rho = pd.concat((rx_carry.iloc[_p::22] for _p in range(22)), axis=1)\
        .corrwith(pd.concat((rx_carry_normal.iloc[_p::22] for _p in range(22)), axis=1))\
        .mean()

    ax.annotate(r"$\bar{\rho} = " + "{:.2f}".format(avg_rho) + "$", 
                xy=(0.05, 0.63), xycoords='axes fraction', 
                xytext=(0.05, 0.63), textcoords='axes fraction')

    fig.tight_layout()
    fig.savefig("reports/figures/carries.png", dpi=200)


if __name__ == '__main__':
    # gen_carry_conv()
    # gen_tsh_example()
    carry_from_conv()
