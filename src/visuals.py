import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator
import seaborn as sns


def plot_convolution_filter(
        m,
        xticklabels=None,
        yticklabels=None,
        figsize=None,
        **heatmap_kwargs
) -> plt.Figure:
    """"""
    t, _ = m.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    sns.heatmap(m, cmap="coolwarm", annot=True, cbar=False, ax=ax, linewidth=0.25,
                fmt=".2f", **heatmap_kwargs)

    # Set the x-axis tick labels
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.yaxis.set_major_locator(IndexLocator(base=1, offset=0.5))
        ax.set_yticklabels(yticklabels, rotation=0)

    ax.set_xlabel("tenor")
    ax.set_ylabel("period")

    return fig, ax


# helper to print dataframes nicely
def style_time_series(df, n_tail=6, mult=1.0, precision=2, **kwargs):
    """
    Parameters
    ----------
    df : pandas.DataFrame
    """
    res = df.tail(n_tail).mul(mult).rename_axis(index="date")
    res = res.style\
        .set_properties(**{'width': '50px'})\
        .format(precision=precision, **kwargs)\
        .format_index(lambda _x: _x.strftime("%Y-%m-%d"))
    return res
