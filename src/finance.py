import warnings

import pandas as pd


def get_long_short_legs(signal, legsize=None):
    """
    Parameters
    ----------
    signal : pd.DataFrame
    legsize : int
        None to do rank-based sorts
    """
    notnull = signal.notnull()

    if legsize is None:
        _long = signal.where(signal.gt(signal.median(axis=1), axis=0)).rank(axis=1)
        _long = _long.div(_long.sum(axis=1), axis=0)
        _short = signal.where(signal.lt(signal.median(axis=1), axis=0))\
            .mul(-1).rank(axis=1)
        _short = _short.div(_short.sum(axis=1), axis=0) * -1
        return _long.fillna(_short)

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

    if not res.sum(axis=1, min_count=1).dropna().eq(0.0).all():
        warnings.warn("Unreliable sorts.")

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
