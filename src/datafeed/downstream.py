import pandas as pd
import numpy as np
from joblib import Memory

from .raw import get_fx_rates, get_fx_fixing_dates

memory = Memory(location="./data")


def mask_as_in_lustig_2008(df):
    """A la lustig et al (2001)."""
    excl = {
        "zar": ("1985-07", "1985-08"),
        "myr": ("1998-08", "2005-06"),
        "idr": ("2000-12", "2007-05"),
        "try": ("1960-01", "2003-07-23"),
        "aed": ("2006-06", "2006-11"),
        "rub": ("1960-01", "2004-08"),
        "ars": ("1960-01", "2002-02"),
        "uyu": ("1960-01", "2009-01"),
        "vnd": ("1960-01", "2007-03"),
        "brl": ("1960-01", "1999-06"),
    }

    for c, dts in excl.items():
        if c in df:
            df.loc[dts[0]:dts[1], [c]] = np.nan

    return df


def mask_eurozone(df):
    """Mask eurozone currencies before 1999-01."""
    eur_subset = ['ats', 'bef', 'dem', 'fim', 'frf', 'grd',
                  'iep', 'itl', 'nlg', 'pte', 'esp', ]

    for c in eur_subset:
        if c in df:
            df.loc["1999-01-04":, [c]] = np.nan

    if "eur" in df:
        df.loc[:"1998-12-30", ["eur"]] = np.nan

    return df


def maturity_str2num(m):
    if m == "spot":
        return 0.0
    if m[-1] == "m":
        return float(m[:-1]) / 12
    elif m[-1] == "y":
        return float(m[:-1])
    elif m[-1] == "w":
        return float(m[:-1]) / 52
    else:
        raise NotImplementedError("unknown maturity literal")


@memory.cache
def get_fx_data() -> dict:
    """Get term structure history and excess returns data.

    Returns a dictionary containing 'term_structure_history' (Tx(MN)
    dataframe of forward rates of xxxusd indexed by dates, columned by
    (xxx, maturity)) and 'excess_returns' (TxN dataframe of forward-looking
    excess return of xxxusd at one given maturity, indexed by dates,
    columned by xxx).

    example 'term_structure_history'
    currency       aud                      ...       jpy
    maturity      0.0       1.0       2.0   ...      9.0       12.0
    1990-01-01  0.7890       NaN       NaN  ...       NaN       NaN
    1990-01-02  0.7865  0.779100  0.774000  ...       NaN  0.006897

    example of 'excess_returns'
    currency       aud     cad     eur     chf  ...     sek
    1990-01-02 -0.0133 -0.0210  0.0265  0.0662  ...  0.0211
    1990-01-03 -0.0103 -0.0209  0.0323  0.0673  ...  0.0274

    """
    # raw data: spot, fwd, fixing dates
    data = get_fx_rates()
    fixing_dt = get_fx_fixing_dates()

    maturities = ['spot', '1w', '2w', '3w', '1m', '2m', '3m', '4m', '5m', '6m', '9m',
                  '12m', '15m', '18m', '2y', '3y', '4y', '5y']
    data["maturity"] = pd.Categorical(data["maturity"],
                                      categories=maturities,
                                      ordered=True)
    ts_history = data \
        .pivot(index="date", columns=["currency", "maturity"], values="price")\
        .sort_index(axis=1)

    ts_history = mask_as_in_lustig_2008(mask_eurozone(ts_history))

    # splice euro
    eur = ts_history["eur"].iloc[::-1].pct_change(fill_method=None) \
        .fillna(ts_history["dem"].loc[:"1998"].iloc[::-1].pct_change()) \
        .add(1) \
        .fillna(pd.concat([c_.dropna().iloc[[-1]] for _, c_ in
                           ts_history["eur"].items()], axis=1)) \
        .cumprod() \
        .iloc[::-1] \
        .where(ts_history["eur"].fillna(ts_history["dem"]).notnull())
    ts_history["eur"] = eur

    spot = ts_history.xs("spot", axis=1, level="maturity")

    # need to convert dates to datetime, but easier to convert unique
    # values, then map
    mapper = pd.Series(pd.to_datetime(fixing_dt["settle_dt"].unique()),
                       index=fixing_dt["settle_dt"].unique())
    fixing_dt["settle_dt"] = fixing_dt["settle_dt"].map(mapper)

    # for each actual date in `fixing_dt` take the (currency, fixing date)
    # pair and look it up in `spot` (make sure every date exists in `spot`)
    spot_at_fix = pd.merge(
        fixing_dt,
        spot.stack().rename("spot").reset_index(),
        left_on=["settle_dt", "currency"], right_on=["date", "currency"]
    ).loc[:, ["date_x", "currency", "maturity", "spot"]]
    spot_at_fix = spot_at_fix.rename(columns={"date_x": "date"}) \
        .pivot(index="date", columns=["currency", "maturity"], values="spot")

    # forward-looking excess and spot returns
    rx = spot_at_fix \
        .div(ts_history.drop("spot", axis=1, level="maturity")) - 1
    rs = spot_at_fix \
        .div(spot, axis=1) - 1
    rs_d = spot.pct_change()

    fd = ts_history.drop("spot", axis=1, level="maturity").pow(-1).mul(
        ts_history.xs("spot", axis=1, level="maturity"),
        axis=1, level=0
    ).sub(1)

    # annualize fd
    ann = fd.columns.get_level_values("maturity").map(maturity_str2num)
    fd = fd.div(ann, axis=1)

    # G10 currencies
    g10 = ["aud", "cad", "eur", "chf", "nzd",
           "gbp", "sek", "nok", "jpy", "dkk"]
    ms = ['spot', '1w', '1m', '2m', '3m', '6m', '9m', '12m', '2y']

    idx = pd.MultiIndex.from_product(
        (g10, pd.Categorical(ms, categories=ms, ordered=True)),
        names=["currency", "maturity"]
    )
    dts = pd.date_range("1990-01-01", "2023-06-30", freq="B")

    # 0.25 and 18 month are illiquid
    res = {
        "term_structure_history": ts_history \
            .reindex(columns=idx, index=dts) \
            .sort_index(axis=1, level=[0, 1]),
        "excess_returns": rx.xs("1m", axis=1, level="maturity") \
                            .reindex(columns=g10, index=dts)\
                            .sort_index(axis=1),
        "spot_returns": rs.xs("1m", axis=1, level="maturity") \
                          .reindex(index=dts)[g10] \
                          .sort_index(axis=1),
        "spot_returns_d": rs_d.reindex(index=dts)[g10] \
                              .sort_index(axis=1),
        "fd_annualized": fd.reindex(index=dts)[g10] \
                           .sort_index(axis=1),
    }

    return res
