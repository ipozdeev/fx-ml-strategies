import numpy as np
import pandas as pd
from functools import reduce

import torch.nn
from torch.utils.data import Dataset


class TermStructureUnitBlockDataset(Dataset):
    """Sample blocks that are the level=0 values.

    One block is e.g. the term structure history for one currency over several periods.

    Parameters
    ----------
    termstructure_df
        wide DataFrame of term structure history, indexed by date, columned by
        (currency, tenor)
    target_df
        wide DataFrame of target variables (e.g. excess returns) indexed by
        date, columned by currency
    lookback : int
        number of time periods to look back
    transform
        function to apply to the features
    """
    def __init__(
            self,
            termstructure_df: pd.DataFrame,
            target_df: pd.DataFrame,
            lookback: int,
            transform: callable = None
    ):
        assert termstructure_df.index.nlevels == 1
        assert termstructure_df.columns.nlevels == 2
        assert target_df.index.nlevels == 1

        self.termstructure_df = termstructure_df
        self.target_df = target_df
        self.lookback = lookback

        n_tenors = len(termstructure_df.columns.unique(level=1))

        if transform is None:
            self.transform = lambda _x: _x
        else:
            self.transform = transform

        # date, currency pairs with no missing observations
        good_dt_ts = termstructure_df.isnull()\
            .rolling(lookback, min_periods=lookback).sum()\
            .groupby(axis=1, level=0).sum(min_count=n_tenors)\
            .stack().eq(0.0)
        good_dt_rx = target_df.notnull().stack()

        self.index = (good_dt_ts & good_dt_rx)\
            .replace(False, np.nan).dropna().index

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, item) -> tuple:
        """"""
        idx = self.index[item]

        x = self.termstructure_df\
            .loc[:idx[0], idx[1]]\
            .tail(self.lookback)\
            .pipe(self.transform)\
            .values\
            .astype(np.float32)

        y = np.array(self.target_df.at[idx[0], idx[1]],
                     dtype=np.float32)

        return x, y


class TermStructureCrossSectionBlockDataset(Dataset):
    """
    Parameters
    ----------
    termstructure_df
        wide DataFrame of term structure history, indexed by date, columned by
        (currency, tenor)
    target_df
        wide DataFrame of target variables (e.g. excess returns) indexed by
        date, columned by currency
    lookback : int
        number of time periods to look back
    transform
        function to apply to the features
    """

    def __init__(
            self,
            termstructure_df: pd.DataFrame,
            lookback,
            transform: callable = None
    ):
        assert termstructure_df.index.nlevels == 1
        assert termstructure_df.columns.nlevels == 2

        self.termstructure_df = termstructure_df
        self.lookback = lookback
        self.lookback_iloc = slice(-lookback, None) if isinstance(lookback, int) else \
            [-_p-1 for _p in sorted(lookback, reverse=True)]

        if transform is None:
            self.transform = lambda _x: _x
        else:
            self.transform = transform

        # date, currency pairs with no missing observations
        if isinstance(lookback, int):
            good_dt_ts = termstructure_df.notnull().all(axis=1) \
                .rolling(lookback, min_periods=lookback).apply(all)\
                .fillna(0.0).astype(int).astype(bool)
        elif hasattr(lookback, "__iter__"):
            all_notnull = termstructure_df.notnull().all(axis=1)
            good_dt_ts = pd.Series(
                {_t: all_notnull.loc[:_t].iloc[self.lookback_iloc].all()
                 for _t in termstructure_df.index[max(lookback):]}
            )
        else:
            raise ValueError("`lookback` must be int or iterable")

        self.index = good_dt_ts.replace(False, np.nan).dropna().index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item) -> tuple:
        """"""
        x = self.termstructure_df \
            .loc[:self.index[item]] \
            .iloc[self.lookback_iloc] \
            .pipe(self.transform) \
            .values \
            .astype(np.float32)

        # y = self.target_df.loc[self.index[item]].values.astype(np.float32)

        return x


class ForwardCrossSectionDataset(Dataset):
    """Sample slices of DataFrame looking forward from (and incl) date t."""

    def __init__(
            self,
            df: pd.DataFrame,
            lookforward: int,
            dropna=False,
            transform=None
    ):
        self.df = df
        self.lookforward = lookforward

        if dropna:
            good_t = df.notnull().all(axis=1) \
                .rolling(lookforward, min_periods=lookforward).apply(all) \
                .shift(-lookforward+1) \
                .fillna(0.0).astype(int).astype(bool)
            self.index = good_t.where(good_t).dropna().index
        else:
            self.index = df.index[:-self.lookforward]

        if transform is None:
            self.transform = torch.nn.Identity()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item) -> tuple:
        x = self.df.loc[self.index[item]:].head(self.lookforward)\
            .values.astype(np.float32)

        x = self.transform(x)

        return x


class MultiDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

        indexes = [_a.index for _a in datasets]

        self.index = reduce(lambda _x, _y: _x.intersection(_y), indexes)

        for _a in self.datasets:
            _a.index = self.index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item) -> tuple:

        res = tuple(_a.__getitem__(item) for _a in self.datasets)

        return res
