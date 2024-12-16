import numpy as np
from pyellispeed import drawing
from pyellispeed import analysis


def test_ellipsoid_dict():
    ell = analysis.Ellipsoid(
            np.array([64, 64, 64]),
            np.array([5, 50, 30]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
    dict_ell = {
            'center': [64, 64, 64],
            'radii': [5, 50, 30],
            'axes': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }
    
    ell_to_dict = analysis.ellipsoid_to_dict(ell)   

    for key in dict_ell:
        if isinstance(dict_ell[key], np.ndarray):
            assert np.array_equal(ell_to_dict[key], dict_ell[key]), f"Mismatch in key '{key}'"
        else:
            assert ell_to_dict[key] == dict_ell[key], f"Mismatch in key '{key}'"

    dict_to_ell = analysis.ellipsoid_from_dict(dict_ell)

    for index, val in enumerate(ell):
        if isinstance(val, np.ndarray):
            assert np.array_equal(dict_to_ell[index], val), f"Mismatch in index '{index}'"
        else:
            assert dict_to_ell[index] == val, f"Mismatch in index '{index}'"


def test_ellipsoid_json():
    ell = analysis.Ellipsoid(
            np.array([64, 64, 64]),
            np.array([5, 50, 30]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
    json_ell = '{"center": [64, 64, 64], "radii": [5, 50, 30], "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}'
    
    ell_to_json = analysis.ellipsoid_to_json(ell)   

    assert ell_to_json == json_ell    

    dict_to_ell = analysis.ellipsoid_from_json(json_ell)

    for index, val in enumerate(ell):
        if isinstance(val, np.ndarray):
            assert np.array_equal(dict_to_ell[index], val), f"Mismatch in index '{index}'"
        else:
            assert dict_to_ell[index] == val, f"Mismatch in index '{index}'"


def test_compute_inertia():
    points = np.array(((9, 10, 10),
                       (11, 10, 10),
                       (10, 8, 10),
                       (10, 12, 10),
                       (10, 10, 9),
                       (10, 10, 11),))
    
    inertia_ellipsoid = analysis.Ellipsoid(
        np.array([10.0,10.0,10.0]),
        np.array([2.52982213, 1.26491106, 1.26491106]),
        np.array([[0., 1., 0.],
                 [0., 0., 1.],
                 [1., 0., 0.]]))

    result = analysis.compute_inertia_ellipsoid(points)
    
    assert np.allclose(result.center, inertia_ellipsoid.center, atol=1e-5), "Centers do not match"
    assert np.allclose(result.radii, inertia_ellipsoid.radii, atol=1e-5), "Radii do not match"
    assert np.allclose(result.axes, inertia_ellipsoid.axes, atol=1e-5), "Axes do not match"

def compute_inertia(angles):
    shape = (128, 128, 128)
    center_xyz = (64, 64, 64)
    radii_xyz = (5, 50, 30)
    angles_xyz = np.deg2rad(angles)

    image = drawing.make_ellipsoid_image(shape, center_xyz, radii_xyz, angles_xyz)
    points = analysis.sample_random_points(image)
    inertial_ellipsoid = analysis.compute_inertia_ellipsoid(points)
    return inertial_ellipsoid.axes


def test_inertia_no_rotation():
    V = compute_inertia([0, 0, 0])

    assert np.linalg.norm(np.abs(V[0]) - [0, 1, 0]) < 1e-1
    assert np.linalg.norm(np.abs(V[1]) - [0, 0, 1]) < 1e-1
    assert np.linalg.norm(np.abs(V[2]) - [1, 0, 0]) < 1e-1


def test_inertia_rotation_around_z():
    V = compute_inertia([0, 0, 45])
    # After rotation around Z (RHS) we expect the following major axis vectors

    # 1st major axis ([0, 1, 0] or [0, -1, 0])
    err1 = np.linalg.norm(V[0] - [-0.70710678, 0.70710678, 0.])
    err2 = np.linalg.norm(V[0] - [0.70710678, -0.70710678, 0.])
    assert min(err1, err2) < 1e-1

    # 2nd Major axis ([0, 0, 1] or [0, 0, -1])
    err1 = np.linalg.norm(V[1] - [0, 0, 1])
    err2 = np.linalg.norm(V[1] - [0, 0, -1])
    assert min(err1, err2) < 1e-1

    # 3rd Major axis ([1, 0, 0] or [-1, 0, 0])
    err1 = np.linalg.norm(V[2] - [0.70710678, 0.70710678, 0.])
    err2 = np.linalg.norm(V[2] - [-0.70710678, -0.70710678, 0.])
    assert min(err1, err2) < 1e-1


def test_inertia_rotation_around_y():
    V = compute_inertia([0, 45, 0])
    # After rotation around Y (RHS) we expect the following major axis vectors
    
    # 1st major axis ([0, 1, 0] or [0, -1, 0])
    err1 = np.linalg.norm(V[0] - [0, 1, 0])
    err2 = np.linalg.norm(V[0] - [0, -1, 0])
    assert min(err1, err2) < 1e-1

    
    # 2nd Major axis ([0, 0, 1] or [0, 0, -1])
    err1 = np.linalg.norm(V[1] - [0.70710678, 0, 0.70710678])
    err2 = np.linalg.norm(V[1] - [-0.70710678, 0, -0.70710678])
    assert min(err1, err2) < 1e-1

    
    # 3rd Major axis ([1, 0, 0] or [-1, 0, 0])
    err1 = np.linalg.norm(V[2] - [0.70710678, 0, -0.70710678])
    err2 = np.linalg.norm(V[2] - [-0.70710678, 0, 0.70710678])
    assert min(err1, err2) < 1e-1


def test_inertia_rotation_around_x():
    V = compute_inertia([45, 0, 0])
    # After rotation around X (RHS) we expect the following major axis vectors
    
    # 1st major axis ([0, 1, 0] or [0, -1, 0])
    err1 = np.linalg.norm(V[0] - [0, 0.70710678, 0.70710678])
    err2 = np.linalg.norm(V[0] - [0, -0.70710678, -0.70710678])
    assert min(err1, err2) < 1e-1

    # 2nd Major axis ([0, 0, 1] or [0, 0, -1])
    err1 = np.linalg.norm(V[1] - [0, 0.70710678, -0.70710678])
    err2 = np.linalg.norm(V[1] - [0, -0.70710678, 0.70710678])
    assert min(err1, err2) < 1e-1

    # 3rd Major axis ([1, 0, 0] or [-1, 0, 0])
    err1 = np.linalg.norm(V[2] - [1, 0, 0])
    err2 = np.linalg.norm(V[2] - [-1, 0, 0])
    assert min(err1, err2) < 1e-1

def test_analyse_empty_sequence():
    analyse = analysis.analyze_sequence([])
    assert len(analyse) == 0, "Analyse of no ellipsoids return should be empty"
    
def test_analyse_sequence_no_rotation():
    ell1 = analysis.Ellipsoid(
        np.array([10.0,10.0,10.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],]))

    ell2 = analysis.Ellipsoid(
        np.array([10.0,11.0,10.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],]))

    ell3 = analysis.Ellipsoid(
        np.array([10.0,12.0,10.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],]))


    analyse = analysis.analyze_sequence([ell1, ell2, ell3])
    
    assert len(analyse) == 1

    rotations = analyse["rotation"]
    assert len(rotations) == 3

    assert rotations[0] == None
    assert rotations[1] == [0.0, 0.0, 0.0]
    assert rotations[2] == [0.0, 0.0, 0.0]

def test_analyse_sequence_no_rotation():
    ell1 = analysis.Ellipsoid(
        np.array([10.0,10.0,10.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],]))

    ell2 = analysis.Ellipsoid(
        np.array([10.0,10.0,10.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([[0., 1., 0.],
                    [0., 0., 1.],
                    [1., 0., 0.],]))

    ell3 = analysis.Ellipsoid(
        np.array([10.0,10.0,10.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([[0.866, -0.5, 0.],
                [0.5, 0.866, 0.],
                [0., 0., 1.]]))


    analyse = analysis.analyze_sequence([ell1, ell2, ell3])
    
    assert len(analyse) == 1

    rotations = analyse["rotation"]
    assert len(rotations) == 3
    
    expected_rotations = [
        None,
        [-np.pi / 2., 0, -np.pi / 2.],
        [np.pi / 3., np.pi / 2., 0]
    ]
    for computed, expected in zip(rotations, expected_rotations):
        if expected is None:
            assert computed is None, "First ellipsoid's rotation should be None"
        else:
            assert np.allclose(computed, expected, atol=1e-5), f"Rotation mismatch: {computed} != {expected}"