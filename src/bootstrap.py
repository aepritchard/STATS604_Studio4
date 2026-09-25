import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def boostrap_sample(data, compute_stat, n_bootstrap=1000):
    """
    Generate the bootstrap distribution of a statistic

    Parameters
    ----------
    data: array-like
        original sample (for regression: 2D array with columns [x,y])

    compute_stat: callable
        function that computes a univariate statistic from data

    n_bootstrap: int, default = 1000
        number of boostrap replicates to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    ValueError
        If data is not 2-dimensional or has wrong shape, n_bootstrap < 1
    TypeError
        If compute_stat is not callable, n_bootstrap is not integer, data is not array-like
    

    Example
    -------
    TBA

    """

    if callable(compute_stat) == False:
        raise TypeError(f"compute_stat must be callable, recieved {type(compute_stat)}")
    
    if isinstance(n_bootstrap, int) == False:
        raise TypeError(f"n_bootstrap must be integer, received {type(n_bootstrap)}") 
    if n_bootstrap < 1:
        raise ValueError(f"Requires n_bootstrap > 1, recieved n_bootstrap = {n_bootstrap}")
    
    try:   
        data_array = np.asarray(data)
    except Exception as error:
        raise TypeError("data must be array-like")

    if data_array.ndim != 2:
        raise ValueError(f"data must be 2-dimensional, but has {data_array.ndim} dimensions")
    if data_array.shape[0] < 1:
        raise ValueError(f"data must have at least one observation, but has {data_array.shape[0]}")
    if data_array.shape[1] != 2:
        raise ValueError(f"data must have exatly 2 columns, but has {data_array.shape[1]}")

    sample_size = len(data)
    bootstrap_stats = [] 

    for i in range(n_bootstrap):
     bootstrap_indices = np.random.choice(range(data.shape[0]), size=sample_size, replace=True)
     bootstrap_stats.append(compute_stat(data[bootstrap_indices]))

    return bootstrap_stats
    


def bootstrap_ci(bootstrap_stats, alpha = 0.05):
    """
    calculate a CI from bootstrap distribution

    Parameters
    ----------
    bootstrap_stats : numpy.ndarray
        bootstrap statisitcs from bootstrap_sample(...)

    alpha : float, default 0.05
        significance level 

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of CI
    
    Raises
    ------
    ValueError
        If alpha not in (0, 1) or if bootstrap_stats is empty

    Example
    -------
        Given a range of stats from 0-1 by 0.1, the 95% CI should be (0.025, 0.975)

    """

    if alpha not in (0,1):
        raise ValueError("alpha must be between 0 and 1")
    if len(bootstrap_stats) == 0:
        raise ValueError("list of bootstrap statistics cannot be empty")

    lower_bound = np.quantile(bootstrap_stats,  alpha / 2)
    upper_bound = np.quantile(bootstrap_stats, 1 - alpha / 2)

    return(lower_bound, upper_bound)



def r_squared(data):
    """
    Calculates R^2 from a linear regression

    Parameters
    ----------
    data : array-like, shape (n, 2)
        Data with columns [x, y]

    Returns
    -------
    float
        R-squared value between 0 and 1

    Raises
    ------
    ValueError
        If data does not have exactly 2 columns or < 2 rows

    """
    if (data.shape) != 2:
            raise ValueError(f"data must be 2-dimensional")
    if data.shape[1] != 2:
        raise ValueError("data must have exactly two columns")
    if data.shape[0] < 2:
        raise ValueError("data must have at least two readings")

    x = data[:, [0]]
    y = data[:, [1]]

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    rsqr = r2_score(y, y_pred)
    return(rsqr)
