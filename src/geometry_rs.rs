use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

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

    Ok(())
}
