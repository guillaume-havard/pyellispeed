from pyellispeed import drawing


def test_bench_make_ellipsoid_image(benchmark):
    benchmark(
        drawing.make_ellipsoid_image, 
        (128, 128, 128), 
        (64, 64, 64), 
        (5, 50, 30), 
        (0.5, 0.7, 0.9)
    )