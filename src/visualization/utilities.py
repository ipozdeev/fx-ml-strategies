import matplotlib.pyplot as plt
import seaborn as sns


def plot_convolution_filter(m, xticklabels=None):
    """"""
    t, _ = m.shape

    _, ax = plt.subplots(figsize=(4, 3))

    # Plot the heatmap
    sns.heatmap(m, cmap="Reds", annot=True, cbar=False, ax=ax, linewidth=0.25, fmt=".2f")

    # Set the y-axis tick labels
    ax.set_yticklabels(['t-{}'.format(i) for i in list(range(t))[::-1]], rotation=0)

    # Set the x-axis tick labels
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    # Show the plot
    plt.show()
