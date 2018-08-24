import numpy as np


def calc_ecdf(a):
    """
    Return the empirical distribution function
    for a given sample.

    Parameters
    ----------
    a : array-like
        Sample

    Returns
    -------
    f : callable
        Empirical distribution function
    """
    a = np.sort(a)
    n = len(a)

    return (lambda x:
            np.searchsorted(a, x, side="right", sorter=None) / n)


def mise(f, g, cmin, cmax, n):
    """
    Estimate mean integrated squared error
    between two functions using Riemann sums.

    Parameters
    ----------
    f : callable
        First function
    g : callable
        Second function
    cmin : int
        Left endpoint
    cmax : int
        Right endpoint
    n : int
        Number of evaluation points

    Returns
    -------
    y : float
        Estimated MISE
    """
    p = np.linspace(cmin, cmax, n, endpoint=False)
    w = (cmax - cmin) / n

    return np.sum([w * (f(i) - g(i))**2 for i in p])


def jackknife(a, func, method="ordinary"):
    """
    Calcualte jackknife estimates for a given sample
    and estimator.

    Parameters
    ----------
    a : array-like
        Sample
    func : callable
        Estimator
    method : string
        * 'ordinary'
        * 'infinitesimal'

    Returns
    -------
    y : np.array
        Jackknife estimates
    """
    n = len(a)
    X = np.reshape(np.delete(np.tile(a, n),
                             [i * n + i for i in range(n)]),
                   newshape=(n, n - 1))

    return np.apply_along_axis(func1d=func,
                               arr=X,
                               axis=1)


def jackknife_bias(a, func, method="ordinary"):
    """
    Calculate jackknife estimate of bias.

    Parameters
    ----------
    a : array-like
        Sample
    func : callable
        Estimator
    method : string
        * 'ordinary'
        * 'infinitesimal'

    Returns
    -------
    y : float
        Jackknife estimate of bias
    """
    return (len(a) - 1) * np.mean(jackknife(a, func, method=method) - func(a))


def jackknife_variance(a, func, method="ordinary"):
    """
    Calculate jackknife estimate of variance.

    Parameters
    ----------
    a : array-like
        Sample
    func : callable
        Estimator
    method : string
        * 'ordinary'
        * 'infinitesimal'

    Returns
    -------
    y : float
        Jackknife estimate of variance
    """
    x = jackknife(a, func, method=method)

    return (len(a) - 1) * np.mean((x - np.mean(x))**2)


def empirical_influence(a, func):
    """
    Calculate the empirical influence function for a given
    sample and estimator using the jackknife method.

    Parameters
    ----------
    a : array-like
        Sample
    func : callable
        Estimator

    Returns
    -------
    y : np.array
        Empirical influence values
    """
    return (len(a) - 1) * (func(a) - jackknife(a, func))
