import numpy as np


def bootstrap(a, func, b, method="ordinary"):
    """
    Calculate estimates from bootstrap samples.

    Parameters
    ----------
    a : array-like
        Original sample
    func : callable
        Estimator to be bootstrapped
    b : int
        Number of bootstrap samples
    method : string
        * 'ordinary'
        * 'balanced'
        * 'antithetic'

    Returns
    -------
    y : np.array
        Estimator applied to each bootstrap sample
    """
    n = len(a)

    if method == "ordinary":
        X = np.reshape(np.random.choice(a, n * b), newshape=(b, n))
    elif method == "balanced":
        X = np.reshape(np.random.permutation(np.repeat(a, b)),
                       newshape=(b, n))
    elif method == "antithetic":
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

    return np.apply_along_axis(func1d=func, arr=X, axis=1)
