import pytest
import numpy as np

from pyellispeed import analysis


@pytest.fixture
def generate_random_big_array():
    """
    Allow to generate a NumPy array with random values.
    """
    def create_array():
        return np.random.rand(100, 3)
    return create_array

@pytest.fixture
def ellipsoids():
    """Generate several ellipsoids"""
    nb_ellipsoids = 100
    return [analysis.Ellipsoid(
                np.random.rand(3) * 128, # 'center'
                np.random.rand(3) * 40, # 'radii'
                np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))) # 'axes'
            ) for _ in range(nb_ellipsoids)]