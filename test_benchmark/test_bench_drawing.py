from pyellispeed import drawing
from pyellispeed import drawing_rs

def test_bench_make_ellipsoid_image_16(benchmark):
    benchmark(
        drawing.make_ellipsoid_image, 
        (16, 16, 16), 
        (8, 8, 8), 
        (1, 5, 3), 
        (0.5, 0.7, 0.9)
    )

def test_bench_make_ellipsoid_image_16_rs(benchmark):
    benchmark(
        drawing_rs.make_ellipsoid_image,
        (16, 16, 16),
        (8., 8., 8.),
        (1., 5., 3.),
        (0.5, 0.7, 0.9)
    )

def test_bench_make_ellipsoid_image_128(benchmark):
    benchmark(
        drawing.make_ellipsoid_image, 
        (128, 128, 128), 
        (64, 64, 64), 
        (5, 50, 30), 
        (0.5, 0.7, 0.9)
    )

def test_bench_make_ellipsoid_image_128_rs(benchmark):
    benchmark(
        drawing_rs.make_ellipsoid_image,
        (128, 128, 128),
        (64., 64., 64.),
        (5., 50., 30.),
        (0.5, 0.7, 0.9)
    )

def test_bench_make_ellipsoid_image_512(benchmark):
    benchmark(
        drawing.make_ellipsoid_image, 
        (512, 512, 512), 
        (256, 256, 256), 
        (50, 200, 100), 
        (0.5, 0.7, 0.9)
    )

def test_bench_make_ellipsoid_image_512_rs(benchmark):
    benchmark(
        drawing_rs.make_ellipsoid_image,
        (512, 512, 512),
        (256., 256., 256.),
        (50., 200., 100.),
        (0.5, 0.7, 0.9)
    )

# Take more than 32GB of RAM
# def test_bench_make_ellipsoid_image_1024(benchmark):
#     benchmark(
#         drawing.make_ellipsoid_image, 
#         (1024, 1024, 1024), 
#         (512, 512, 512), 
#         (50, 500, 300), 
#         (0.5, 0.7, 0.9)
#     )

def test_bench_make_ellipsoid_image_1024_rs(benchmark):
    benchmark(
        drawing_rs.make_ellipsoid_image, 
        (1024, 1024, 1024), 
        (512, 512, 512), 
        (50, 500, 300), 
        (0.5, 0.7, 0.9)
    )