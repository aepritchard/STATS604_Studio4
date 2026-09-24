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
        raise TypeError(f"n_bootstrap must be an integrer, received {type(n_bootstrap)}") 
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
     bootstrap_indices = np.random.choice(data, size=sample_size, replace=True)
     bootstrap_stats.append(compute_stat(data[bootstrap_indices]))

    return bootstrap_stats
    


def boostrap_ci(boostrap_Stats, alpha = 0.05):
    """
    calculate a CI from bootstrap distribution

    Parameters
    ----------
    boostrap_stats : numpy.ndarray
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
    TBA

    """

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