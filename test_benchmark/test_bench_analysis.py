import numpy as np
from pyellispeed import analysis

def test_bench_ellipsoid_to_dict(benchmark):
    benchmark(
        analysis.ellipsoid_to_dict, 
        analysis.Ellipsoid(
            (64, 64, 64),
            (5, 50, 30),
            np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        )
    )

def test_bench_ellipsoid_from_dict(benchmark):
    benchmark(
        analysis.ellipsoid_from_dict, 
        {
            'center': (64, 64, 64),
            'radii': (5, 50, 30),
            'axes': ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        }
    )

def test_bench_ellipsoid_to_json(benchmark):
    benchmark(
        analysis.ellipsoid_to_json, 
        analysis.Ellipsoid(
            (64, 64, 64),
            (5, 50, 30),
            np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        )
    )

def test_bench_ellipsoid_from_json(benchmark):
    benchmark(
        analysis.ellipsoid_from_json, 
        '{"center": [64, 64, 64], "radii": [5, 50, 30], "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}'
    )

def test_bench_sample_random_points(benchmark):
    image =np.random.randint(0, 3, (128, 128, 128))
    n = 2000
    benchmark(analysis.sample_random_points, image, n)
    
def test_bench_compute_inertia_ellipsoid(benchmark):
    points = np.random.rand(2000, 3) * 128
    benchmark(analysis.compute_inertia_ellipsoid, points)


def test_bench_analyze_sequence(benchmark, ellipsoids):
    benchmark(analysis.analyze_sequence, ellipsoids)