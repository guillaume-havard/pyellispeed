use super::geometry_rs;
use ndarray::{Array1, Zip};
use numpy::{PyArray3, PyArrayMethods};
use pyo3::prelude::*;

pub fn make_ellipsoid_image<'py>(
    py: Python<'py>,
    shape: [usize; 3],
    center: [f64; 3],
    radii: [f64; 3],
    angles: [f64; 3],
) -> Bound<'py, PyArray3<bool>> {
    let rotation = geometry_rs::build_rotation_matrix(angles[0], angles[1], angles[2], Some(false));

    // Generate a 3D grid of coordinates
    let ellipsoid = PyArray3::<bool>::zeros(py, shape, false);
    let (cx, cy, cz) = (center[0], center[1], center[2]);
    let radii = Array1::from(vec![radii[0], radii[1], radii[2]]);
    // let radii_sq = Array1::from_vec(vec![radii[0]*radii[0], radii[1]*radii[1], radii[2]*radii[2]]);

    // This is safe because the PyArray `ellipsoid` is already allocated in Python's memory,
    // and the returned reference is tied to the same lifetime as py (the Python interpreter's lifetime).
    unsafe {
        let mut ellipsoid_mut = ellipsoid.as_array_mut();

        Zip::indexed(&mut ellipsoid_mut).par_for_each(|(z, y, x), val| {
            let coords = Array1::from_vec(vec![x as f64 - cx, y as f64 - cy, z as f64 - cz]);

            let coords_prime = rotation.dot(&coords);

            // equation = x_prime.powi(2) / rx.powi(2) + y_prime.powi(2) / ry.powi(2) + z_prime.powi(2) / rz.powi(2);
            let equation = &coords_prime.mapv(|x| x * x) / &radii.mapv(|x| x * x);
            let equation = equation.sum();

            // Check if the point is on the ellipsoid
            *val = equation == 1.0;
        });

        ellipsoid
    }
}

#[pymodule]
pub fn drawing_rs<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "make_ellipsoid_image")]
    pub fn make_ellipsoid_image_rs<'py>(
        py: Python<'py>,
        shape: [usize; 3],
        center: [f64; 3],
        radii: [f64; 3],
        angles: [f64; 3],
    ) -> Bound<'py, PyArray3<bool>> {
        make_ellipsoid_image(py, shape, center, radii, angles)
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_ellipsoid_image() {
        let shape = [100, 100, 100];
        let center_xyz = [50., 50., 50.];
        let radii_xyz = [5., 10., 30.];
        let angles_xyz = [0., 0., 0.];

        let image = make_ellipsoid_image(Py, shape, center_xyz, radii_xyz, angles_xyz);
        unsafe {
            let image = image.as_array();
        
            assert_eq!(image.shape(), shape);
            let nonzero_count = image.iter().filter(|&&x| x).count();
            assert_ne!(nonzero_count, 0);

            assert_eq!(image[[20, 50, 50]], true);
            assert_eq!(image[[80, 50, 50]], true);
            assert_eq!(image[[50, 40, 50]], true);
            assert_eq!(image[[50, 60, 50]], true);
            assert_eq!(image[[50, 50, 45]], true);
            assert_eq!(image[[50, 50, 55]], true);

            //assert_eq!(image[[50, 50, 54]] , false);
            assert_eq!(image[[50, 50, 56]], false);
            assert_eq!(image[[50, 49, 55]], false);
            assert_eq!(image[[50, 51, 55]], false);
            assert_eq!(image[[49, 50, 55]], false);
            assert_eq!(image[[51, 50, 55]], false);
        }
    }
}
