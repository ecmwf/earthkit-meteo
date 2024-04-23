import numpy as np


def crps(x, y):
    """Computes Continuous Ranked Probability Score (CRPS).

    Parameters
    ----------
    x: numpy array (n_ens, n_points)
        Ensemble forecast
    y: numpy array (n_points)
        Observation/analysis

    Returns
    -------
    numpy array (n_points)
        CRPS values


    The method is described in [Hersbach2000]_.
    """
    # first sort ensemble
    x.sort(axis=0)

    # construct alpha and beta, size nens+1
    n_ens = x.shape[0]
    shape = (n_ens + 1,) + x.shape[1:]
    alpha = np.zeros(shape)
    beta = np.zeros(shape)

    # x[i+1]-x[i] and x[i]-y[i] arrays
    diffxy = x - y.reshape(1, *(y.shape))
    diffxx = x[1:] - x[:-1]  # x[i+1]-x[i], size ens-1

    # if i == 0
    alpha[0] = 0
    beta[0] = np.fmax(diffxy[0], 0)  # x(0)-y
    # if i == n_ens
    alpha[-1] = np.fmax(-diffxy[-1], 0)  # y-x(n)
    beta[-1] = 0
    # else
    alpha[1:-1] = np.fmin(
        diffxx, np.fmax(-diffxy[:-1], 0)
    )  # x(i+1)-x(i) or y-x(i) or 0
    beta[1:-1] = np.fmin(diffxx, np.fmax(diffxy[1:], 0))  # 0 or x(i+1)-y or x(i+1)-x(i)

    # compute crps
    p_exp = (np.arange(n_ens + 1) / float(n_ens)).reshape(n_ens + 1, *([1] * y.ndim))
    crps = np.sum(alpha * (p_exp**2) + beta * ((1 - p_exp) ** 2), axis=0)

    return crps
