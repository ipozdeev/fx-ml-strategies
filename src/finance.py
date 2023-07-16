import pandas as pd


def get_long_short_legs(signal, legsize):
    """"""
    notnull = signal.notnull()

    # insert median where there are nans to be able to rank (to be dropped)
    med = signal.median(axis=1)
    for _c, _col in signal.items():
        signal.loc[:, _c] = _col.fillna(med)

    # rank
    rnk = signal.rank(axis=1, pct=True)

    # sort
    n_assets = signal.shape[1]
    res = pd.DataFrame().reindex_like(rnk).fillna(0.0)
    res = res\
        .mask(rnk <= legsize/n_assets, -1.0)\
        .mask(rnk > (1 - legsize/n_assets), +1.0)\
        .where(notnull)

    assert res.sum(axis=1, min_count=1).dropna().eq(0.0).all()

    n_short = res.where(res < 0).count(axis=1)
    n_long = res.where(res > 0).count(axis=1)

    res = res\
        .mask(res < 0, res.div(n_short, axis=0))\
        .mask(res > 0, res.div(n_long, axis=0))

    return res


if __name__ == '__main__':
    import numpy as np
    sig = pd.DataFrame(np.random.normal(size=(10, 5)))
    sig.iloc[0, 2:] = np.nan
    sig.iloc[-1, -2:] = np.nan
    get_long_short_legs(sig, 2)
