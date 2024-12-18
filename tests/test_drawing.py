import numpy as np
from pyellispeed import drawing
from pyellispeed import drawing_rs

def test_make_ellipsoid_image():
    shape = (100, 100, 100)# Z, Y, X
    center_xyz = (50, 50, 50)
    radii_xyz = (5, 10, 30)
    angles_xyz = (0, 0, 0) # X, Y, Z

    image = drawing_rs.make_ellipsoid_image(shape, center_xyz, radii_xyz, angles_xyz)
    assert image.shape == shape
    assert np.count_nonzero(image) > 0

    assert image[20, 50, 50] == 1
    assert image[80, 50, 50] == 1
    assert image[50, 40, 50] == 1
    assert image[50, 60, 50] == 1
    assert image[50, 50, 45] == 1
    assert image[50, 50, 55] == 1

    #assert image[50, 50, 54] == 0
    assert image[50, 50, 56] == 0
    assert image[50, 49, 55] == 0
    assert image[50, 51, 55] == 0
    assert image[49, 50, 55] == 0
    assert image[51, 50, 55] == 0

def test_make_ellipsoid_image_rotation():
    shape = (100, 100, 100)# Z, Y, X
    center_xyz = (50, 50, 50)
    radii_xyz = (5, 10, 30)
    angles_xyz = (0, np.pi/2, 0) # X, Y, Z

    image = drawing_rs.make_ellipsoid_image(shape, center_xyz, radii_xyz, angles_xyz)
    assert image.shape == shape
    assert np.count_nonzero(image) > 0

    assert image[45, 50, 50] == 1
    assert image[55, 50, 50] == 1
    assert image[50, 40, 50] == 1
    assert image[50, 60, 50] == 1
    assert image[50, 50, 20] == 1
    assert image[50, 50, 80] == 1

    # assert image[50, 50, 79] == 0
    assert image[50, 50, 81] == 0
    assert image[50, 49, 80] == 0
    assert image[50, 51, 80] == 0
    assert image[49, 50, 80] == 0
    assert image[51, 50, 80] == 0
