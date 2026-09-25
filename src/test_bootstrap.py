import pytest
import numpy as np
import pandas as pd
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared


def test_bootstrap_sample_happy_path():
    """
    test base functionality
    given two perfectly correlated vectors, their bootstrap r_squared estimates should be very close to one
    """
    v1 = np.arange(20)
    v2 = 3*v1
    data = np.column_stack((v1, v2))

    bootstrap_stats = bootstrap_sample(data, r_squared, 10)
    assert(all([x - 1 for x in bootstrap_stats]) < 1e-6)


def test_bootstrap_sample_input_errors():
    data = np.array([[1,2],[2,3.9],[3,5.8]])
    with pytest.raises(TypeError, match = "must be callable"):
        bootstrap_sample(data, "add")
    with pytest.raises(TypeError, match = "must be callable"):
        bootstrap_sample(data, [5,3,2])
    with pytest.raises(TypeError, match = "must be callable"):
        bootstrap_sample(data, True)

    with pytest.raises(TypeError, match = "must be integer"):
        bootstrap_sample(data, r_squared, 3.2)
    with pytest.raises(TypeError, match = "must be integer"):
        bootstrap_sample(data, r_squared, "abc")

    with pytest.raises(TypeError, match = "must be array-like"):
        bootstrap_sample((1,2,3), r_squared)
    with pytest.raises(TypeError, match = "must be array-like"):
        bootstrap_sample("abcd", r_squared)
    with pytest.raises(TypeError, match = "must be array-like"):
        bootstrap_sample(False, r_squared)

    with pytest.raises(ValueError, match = "must be 2-dimensional"):
        bootstrap_sample((1,2,3), r_squared)
    with pytest.raises(ValueError, match = "must be 2-dimensional"):
        bootstrap_sample(np.arange(24).reshape(2,3,4), r_squared)
    with pytest.raises(ValueError, match = "must be 2-dimensional"):
        bootstrap_sample(np.arange(24).reshape(1,2,3,4), r_squared)

    with pytest.raises(ValueError, match = "must have at least one observation"):
        bootstrap_sample(pd.DataFrame(columns = ['x', 'y']), r_squared)
    with pytest.raises(ValueError, match = "must have at least one observation"):
            bootstrap_sample(pd.DataFrame(columns = ['a', 'b', 'c']), r_squared)
        
    
    with pytest.raises(ValueError, match = "must have exactly 2 columns"):
            bootstrap_sample(np.arange(24).reshape(4,6), r_squared)
    
    with pytest.raises(ValueError, match = "must have exactly 2 columns"):
            bootstrap_sample(np.arange(24).reshape(3,8), r_squared)
    with pytest.raises(ValueError, match = "must have exactly 2 columns"):
        bootstrap_sample(np.arange(6).reshape(6,), r_squared)


def test_bootstrapCI_happy_path():
    """
    test base functionality
    given a range of stats from 0-1 by 0.1, the 95% CI should be (0.025, 0.975)
    """
    bootstrap_stats = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    alpha = 0.05

    bootstrap_CI = bootstrap_ci(bootstrap_stats = bootstrap_stats, alpha = 0.05)
    assert abs(bootstrap_CI[0] - 0.025) < 1e-10
    assert abs(bootstrap_CI[1] - 0.975) < 1e-10

def test_bootstrapCI_input_errors():
    """
    tests if incorrect input raises the correct errors
    """
    bootstrap_stats = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    with pytest.raises(ValueError, match="list of bootstrap statistics cannot be empty"):
        bootstrap_ci(bootstrap_stats = [], alpha = 0.5)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci(bootstrap_stats = bootstrap_stats, alpha = 1)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci(bootstrap_stats = bootstrap_stats, alpha = 0)

def test_r_squared_happy_path():
    """
    test base functionality
    x = [1,2,3,4,5] and y=[2,4,6,8,10] should return R-squared=1
    """
    data = np.column_stack(([1,2,3,4,5], [2,4,6,8,10]))
    assert(r_squared(data) - 1) < 1e-10

def test_r_squared_input_errors():
    """
    test if incorrect input raises the correct errors
    """
    with pytest.raises(ValueError, match="data must be 2-dimensional"):
        r_squared(data=[])
    with pytest.raises(ValueError, match="data must be 2-dimensional"):
        r_squared(data=[1,2,3,4])
    with pytest.raises(ValueError, match="data must have at least two readings"):
        r_squared(data=np.column_stack(([1], [2])))
    with pytest.raises(ValueError, match="data must have exactly two columns"):
        r_squared(data=np.column_stack(([1,1,1], [2,2,2], [3,3,3])))

