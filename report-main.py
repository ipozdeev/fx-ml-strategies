import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datafeed.dataloader import (
    TermStructureCrossSectionBlockDataset,
    ForwardCrossSectionDataset,
    MultiDataset
)
from src.datafeed.downstream import get_fx_data, maturity_str2num
from src.losses import custom_loss_return
from src.models import ConvNet
from src.visuals import plot_convolution_filter


def train_loop(model, loss_func, dataloader, optimizer) -> float:
    """Train model for one epoch.

    Assumes `loss_func` returns an average loss across batches.
    """
    # init loss for this epoch
    loss_value = 0.0

    for _batch, (_x, _y) in enumerate(dataloader):
        # _x is (batch, lookback, assets*tenors), but conv2d needs channels
        y_hat = model(_x.unsqueeze(1))

        # y_hat is of dim (batch, 1, assets)
        loss = loss_func(y_hat, _y)

        # back-propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        loss_value -= loss.item()

        if _batch % 40 == 0:
            loss_now, batch_now = loss_value / (_batch + 1) * 1200, _batch * _x.shape[0]
            print(
                "avg rx: {: .2f} | [{:>5d}/{:>5d}]" \
                    .format(loss_now, batch_now, len(dataloader.dataset))
            )

    print("epoch done!")

    return loss_value / (_batch + 1) * 1200


def test_loop(model, x_test_batched, target_rx) -> float:
    """Test the model.

    Parameters
    ----------
    model : torch.nn.Module
    x_test_batched : torch.Tensor
        covering the whole test sample, one batch per period (expected to
        have lots of batches)
    target_rx : pandas.DataFrame
        od out-of-sample rx
    """
    with torch.no_grad():
        y_hat = model(x_test_batched.unsqueeze(1))

    y_hat = pd.DataFrame(
        y_hat[:, 0, :].numpy(),
        index=target_rx.index,
        columns=target_rx.columns
    )

    y_hat_rx = target_rx.mul(y_hat).sum(axis=1, min_count=1).mean() * 1200

    print("--")
    print("avg rx oos: {: .2f}%".format(y_hat_rx))

    return y_hat_rx


def train_many_models():
    """"""
    # parameters
    train_smpl_end = pd.to_datetime("2009-12-31")
    test_smpl_start = train_smpl_end + Day()
    lookback = (
            [0, 1, 2, 3, 4, 5] +
            list(np.round(np.linspace(22, 252, num=12), 0).astype(int))
    )
    if isinstance(lookback, int):
        lookback_list = list(range(lookback))[::-1]
    else:
        lookback_list = lookback

    epochs = 40
    lr = 1e-03

    # data
    data = get_fx_data()

    # drop illiquid contracts, ffill a lil'
    ts = (data["term_structure_history"].drop(["2y", "1w"], axis=1, level=1) \
          .ffill(limit=2))
    rx = data["excess_returns"]

    tenors = list(
        ts.columns.get_level_values("maturity").remove_unused_categories().categories
    )
    n_assets, n_tenors = rx.shape[1], len(tenors)

    # annualizer of forward discounts, tbu in dataloaders
    fd_scaler = pd.Series(
        data=ts.columns.unique("maturity").map(maturity_str2num).astype(float),
        index=ts.columns.unique("maturity").astype(str),
        name="maturity"
    )
    fd_scaler.loc["spot"] = 1.0

    def transform_fwd_premium(_p):
        _res = _p \
            .sub(_p.iloc[-1].xs("spot", level="maturity"), axis=1, level="currency") \
            .div(fd_scaler, axis=1, level="maturity")
        return _res

    # features: TSH
    ds_back = TermStructureCrossSectionBlockDataset(
        np.log(ts.loc[:train_smpl_end]),
        lookback=lookback,
        transform=transform_fwd_premium
    )

    # excess returns for calculation of the sharpe ratio numerator
    ds_fwd_rx = ForwardCrossSectionDataset(
        rx.loc[:train_smpl_end],
        lookforward=1,
        dropna=True
    )
    ds_train = MultiDataset(ds_back, ds_fwd_rx)

    dl_train = DataLoader(ds_train, batch_size=10, shuffle=True)

    ds_test = TermStructureCrossSectionBlockDataset(
        np.log(ts.loc[test_smpl_start:]),
        lookback=lookback,
        transform=transform_fwd_premium
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=len(ds_test),
        shuffle=False
    )
    x_test = next(iter(dl_test))

    for _m in range(4):
        m = ConvNet(lookback if isinstance(lookback, int) else len(lookback),
                    n_tenors, init_weights="normal")

        opt = torch.optim.Adam(m.parameters(), lr=1e-03)

        rx_ins = []
        rx_oos = []

        # pre-training values
        _rx_i = train_loop(m, custom_loss_return, dl_train, opt)
        _rx_o = test_loop(m, x_test_batched=x_test, target_rx=rx.loc[ds_test.index])
        rx_ins.append(_rx_i)
        rx_oos.append(_rx_o)

        for _e in range(epochs):
            print(f"doing epoch {_e}")
            _rx_i = train_loop(m, custom_loss_return, dl_train, opt)
            _rx_o = test_loop(m, x_test_batched=x_test, target_rx=rx.loc[ds_test.index])
            rx_ins.append(_rx_i)
            rx_oos.append(_rx_o)

            if _e == 15:
                torch.save(m.state_dict(), f'models/m-{_m}-conv.pth')

                fig = plot_convolution_filter(
                    m.conv_1.weight.detach().numpy()[0, 0, :, :],
                    xticklabels=tenors,
                    yticklabels=['t-{}'.format(i) for i in lookback_list[::-1]],
                    figsize=(5, 2.5),
                    center=0.0
                )
                fig.savefig(f"figures/m-{_m}-conv-post.png", dpi=200)

            pd.DataFrame({"ins": rx_ins, "oos": rx_oos})\
                .to_csv(f"data/output/rx-ins-oos-m{_m}.csv")


if __name__ == '__main__':
    train_many_models()
