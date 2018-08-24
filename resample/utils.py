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


def bootstrap(a, func=None, b=100, method="ordinary"):
    """
    Calculate estimates from bootstrap samples.

    Parameters
    ----------
    a : array-like
        Original sample
    func : callable or None
        Estimator to be bootstrapped, data set
        is return if None
    b : int
        Number of bootstrap samples
    method : string
        * 'ordinary'
        * 'balanced'
        * 'antithetic'

    Returns
    -------
    y | X : np.array
        Estimator applied to each bootstrap sample,
        or bootstrap samples if func is None
    """
    a = np.asarray(a)
    n = len(a)

    if method == "ordinary":
        X = np.reshape(np.random.choice(a, n * b), newshape=(b, n))
    elif method == "balanced":
        X = np.reshape(np.random.permutation(np.repeat(a, b)),
                       newshape=(b, n))
    elif method == "antithetic":
        if func is None:
            raise ValueError("func cannot be None when"
                             " method is 'antithetic'")
        indx = np.argsort(empirical_influence(a, func))
        indx_arr = np.reshape(np.random.choice(indx, size=b // 2 * n),
                              newshape=(b // 2, n))
        n_arr = np.full(shape=(b // 2, n), fill_value=n - 1)
        X = a[np.vstack((indx_arr, n_arr - indx_arr))]
    else:
        raise ValueError("method must be either 'ordinary'"
                         " , 'balanced', or 'antithetic',"
                         " '{method}' was"
                         " supplied".format(method=method))

    if func is None:
        return X
    else:
        return np.apply_along_axis(func1d=func, arr=X, axis=1)
