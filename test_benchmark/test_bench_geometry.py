import numpy as np
from pyellispeed import geometry
from pyellispeed import geometry_rs

def test_bench_build_rotation_matrix(benchmark):
    benchmark(geometry.build_rotation_matrix, 10.0, 20.0, 30.0, True)

def test_bench_build_rotation_matrix_rs(benchmark):
    benchmark(geometry_rs.build_rotation_matrix, 10.0, 20.0, 30.0, True)

def test_bench_rotation_matrix_to_angle(benchmark):    
    rotation_matrix = np.array([[0, 0, 1], [0, 0, 2], [1, 2, 3]])

    benchmark(geometry.rotation_matrix_to_angles, rotation_matrix)

def test_bench_rotation_matrix_to_angle_z_0(benchmark):    
    rotation_matrix = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])

    benchmark(geometry.rotation_matrix_to_angles, rotation_matrix)

def test_bench_find_relative_vector_rotation(benchmark):
    benchmark(geometry.find_relative_vector_rotation, np.array([1, 2, 3]), np.array([3, 2, 1]))

def test_bench_find_relative_axes_rotation(benchmark):
    original_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rot_angles = np.deg2rad([0, 0, 45]) # Rotate around Z-axis by 45 deg

    # Build rotation matrix and rotate the axes
    rotm = geometry.build_rotation_matrix(*rot_angles)
    rotated_axes = np.dot(rotm, np.transpose(original_axes)).T

    benchmark(geometry.find_relative_axes_rotation, original_axes, rotated_axes, True)

def test_bench_scalar_projection(benchmark):
    benchmark(geometry.scalar_projection, np.array([1, 2, 3]), np.array([3, 2, 1])) 

def test_bench_find_vectors_mapping(benchmark, generate_random_big_array):
    benchmark(geometry.find_vectors_mapping, 
              generate_random_big_array(), 
              generate_random_big_array()) 