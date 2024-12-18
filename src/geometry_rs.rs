use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::f64::consts::PI;

pub fn build_rotation_matrix<'py>(ax: f64, ay: f64, az: f64, inverse: Option<bool>) -> Array2<f64> {
    let (ax, ay, az) = if inverse.unwrap_or(false) {
        (-ax, -ay, -az)
    } else {
        (ax, ay, az)
    };

    let cos_ax = ax.cos();
    let sin_ax = ax.sin();
    let cos_ay = ay.cos();
    let sin_ay = ay.sin();
    let cos_az = az.cos();
    let sin_az = az.sin();

    let rx = array![
        [1.0, 0.0, 0.0],
        [0.0, cos_ax, -sin_ax],
        [0.0, sin_ax, cos_ax]
    ];

    let ry = array![
        [cos_ay, 0.0, sin_ay],
        [0.0, 1.0, 0.0],
        [-sin_ay, 0.0, cos_ay]
    ];

    let rz = array![
        [cos_az, -sin_az, 0.0],
        [sin_az, cos_az, 0.0],
        [0.0, 0.0, 1.0]
    ];

    let rotation = rz.dot(&ry).dot(&rx);

    rotation
}

pub fn rotation_matrix_to_angles(rotation_matrix: ArrayView2<f64>) -> Array1<f64> {
    let eps = 1e-6; // Small threshold for floating-point comparisons

    // Check if R[2, 0] is close to -1 or 1
    if (rotation_matrix[[2, 0]] - 1.0).abs() > eps && (rotation_matrix[[2, 0]] + 1.0).abs() > eps {
        // Generotation_matrixal case
        let ay = -rotation_matrix[[2, 0]].asin();
        let c = ay.cos();
        let ax = (rotation_matrix[[2, 1]] / c).atan2(rotation_matrix[[2, 2]] / c);
        let az = (rotation_matrix[[1, 0]] / c).atan2(rotation_matrix[[0, 0]] / c);
        Array1::from_vec(vec![ax, ay, az])
    } else {
        // Special case: ay = +/- PI/2
        let az = 0.0;
        let ay;
        let ax;

        if (rotation_matrix[[2, 0]] - (-1.0)).abs() < eps {
            // rotation_matrix[2, 0] == -1 => ay = PI / 2
            ay = PI / 2.0;
            ax = az + (rotation_matrix[[0, 1]]).atan2(rotation_matrix[[0, 2]]);
        } else {
            // rotation_matrix[2, 0] == 1 => ay = -PI / 2
            ay = -PI / 2.0;
            ax = -az + (-rotation_matrix[[0, 1]]).atan2(-rotation_matrix[[0, 2]]);
        }

        Array1::from_vec(vec![ax, ay, az])
    }
}


#[pymodule]
pub fn geometry_rs<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (ax, ay, az, inverse=false), name = "build_rotation_matrix")]
    pub fn build_rotation_matrix_rs<'py>(
        py: Python<'py>,
        ax: f64,
        ay: f64,
        az: f64,
        inverse: Option<bool>,
    ) -> Bound<'py, PyArray2<f64>> {
        build_rotation_matrix(ax, ay, az, inverse).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "rotation_matrix_to_angles")]
    pub fn rotation_matrix_to_angles_rs<'py>(
        py: Python<'py>,
        rotation_matrix: PyReadonlyArray2<'py, f64>
    ) -> Bound<'py, PyArray1<f64>> {
        rotation_matrix_to_angles(rotation_matrix.as_array()).into_pyarray(py)
    }

    Ok(())
}
