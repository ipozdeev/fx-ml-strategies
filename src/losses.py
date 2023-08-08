import torch


def custom_loss_sharpe(w_hat, vcv, rx):
    """Custom loss function, the Sharpe ratio.

    Parameters
    ----------
    w_hat : Tensor
        portfolio weights, of shape (batches, 1, assets)
    vcv : Tensor
        covariance matrices, of dim (batches, assets, assets)
    rx : Tensor

    Returns
    -------
    Tensor
        scalar
    """
    qf = w_hat @ vcv @ w_hat.transpose(1, 2)
    _res = w_hat @ rx.transpose(1, 2) / torch.sqrt(qf)
    _res = -1 * _res.mean()
    return _res


def custom_loss_return(w_hat, rx):
    """Custom loss function, simply the return.

    Parameters
    ----------
    w_hat : Tensor
        portfolio weights, of shape (batches, 1, assets)
    rx : Tensor
        returns of individual assets

    Returns
    -------
    Tensor
        scalar

    """
    res = w_hat @ rx.transpose(1, 2)
    res = -1 * res.mean()
    return res
