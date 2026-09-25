import pytest
import numpy as np
import pandas as pd
from demo import calculate_correlation

def test_bootstrapCI_happy_path():
    """
    test base functionality
    given a range of stats from 0-1 by 0.1, the 95% CI should be (0.025, 0.975)
    """
    bootstrap_stats = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    alpha = 0.05

    bootstrap_CI = boostrap_ci(boostrap_Stats = bootstrap_stats, alpha = 0.05)
    assert abs(bootstrap_CI[0] - 0.025) < 1e-10
    assert abs(bootstrap_CI[1] - 0.975) < 1e-10

def test_bootstrapCI_input_errors():
    """
    tests if incorrect input raises the correct errors
    """
    bootstrap_stats = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    with pytest.raises(ValueError, match=""):
        bootstrap_ci(boostrap_Stats = [], alpha = 0.5)
    with pytest.raises(ValueError, match=""):
        bootstrap_ci(boostrap_Stats = bootstrap_stats, alpha = 1)
    with pytest.raises(ValueError, match=""):
        bootstrap_ci(boostrap_Stats = bootstrap_stats, alpha = 0)

def test_r_squared_happy_path():
    """
    test base functionality
    x = [1,2,3,4,5] and y=[2,4,6,8,10] should return R-squared=1
    """
    data = np.column_stack([1,2,3,4,5], [2,4,6,8,10])
    assert(r_squared(data) - 1) < 1e-10

def test_r_squared_input_errors():
    """
    test if incorrect input raises the correct errors
    """
    with pytest.raises(ValueError, match=""):
        r_squared(data=[])
    with pytest.raises(ValueError, match=""):
        r_squared(data=[1,2,3,4])
    with pytest.raises(ValueError, match=""):
        r_squared(data=np.column_stack([1], [2]))
    with pytest.raises(ValueError, match=""):
        r_squared(data=np.column_stack([1,1,1], [2,2,2], [3,3,3]))
