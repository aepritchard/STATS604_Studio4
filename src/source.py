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
        If data is empty, n_bootstrap < 1, or data has wrong shape
    TypeError
        If compute_stat is not callable
    

    Example
    -------
    TBA

    """

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