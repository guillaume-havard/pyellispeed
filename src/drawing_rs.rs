use super::geometry_rs;
use ndarray::prelude::*;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;

fn make_ellipsoid_image(
    shape: [usize; 3],
    center: [f64; 3],
    radii: [f64; 3],
    angles: [f64; 3],
) -> Array3<u8> {
    let neg_angles: [f64; 3] = [-angles[0], -angles[1], -angles[2]];
    let rotation_matrix =
        geometry_rs::build_rotation_matrix(neg_angles[0], neg_angles[1], neg_angles[2], Some(false));

    // Create the grid for all three dimensions
    let xi: Vec<Array1<f64>> = shape
        .iter()
        .map(|&s| Array::linspace(0.0, s as f64 - 1.0, s).mapv(|v| v - (s as f64 / 2.0).floor()))
        .collect();

    // Create the meshgrid as a flat matrix
    let grid_points = Array::from_shape_fn((3, shape[0] * shape[1] * shape[2]), |(dim, idx)| {
        let z = idx / (shape[1] * shape[2]);
        let y = (idx / shape[2]) % shape[1];
        let x = idx % shape[2];
        let coord = match dim {
            0 => xi[2][x],
            1 => xi[1][y],
            2 => xi[0][z],
            _ => unreachable!(),
        };
        coord
    });

    // Rotate the grid points
    let rotated_points = rotation_matrix.dot(&grid_points);

    // Compute the grid center in rotated space
    let grid_center: Array1<f64> = center
        .iter()
        .zip(shape.iter().rev())
        .map(|(&c, &s)| c - (s as f64 / 2.0))
        .collect::<Array1<f64>>()
        .dot(&rotation_matrix);

    // Compute the ellipsoid mask in parallel
    let radii_sq: Array1<f64> = radii.iter().rev().map(|&r| r * r).collect();
    let grid_center_rev: Array1<f64> = grid_center.iter().rev().cloned().collect();

    let mut ellipsoid = Array3::<u8>::zeros(shape);
    let shape_prod = shape[1] * shape[2];

    // Use par_map_inplace to modify the array in parallel
    ellipsoid.indexed_iter_mut().for_each(|((z, y, x), val)| {
        let idx = z * shape_prod + y * shape[2] + x;
        let dx = rotated_points[[2, idx]] - grid_center_rev[0];
        let dy = rotated_points[[1, idx]] - grid_center_rev[1];
        let dz = rotated_points[[0, idx]] - grid_center_rev[2];
        let value = (dx * dx) / radii_sq[0] + (dy * dy) / radii_sq[1] + (dz * dz) / radii_sq[2];
        if value <= 1.0 {
            *val = 1;
        }
    });

    ellipsoid
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
    ) -> Bound<'py, PyArray3<u8>> {
        make_ellipsoid_image(shape, center, radii, angles).into_pyarray(py)
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

        let image = make_ellipsoid_image(shape, center_xyz, radii_xyz, angles_xyz);
        assert_eq!(image.shape(), shape);
        let nonzero_count = image.iter().filter(|&&x| x > 0).count();
        assert_ne!(nonzero_count, 0);

        assert_eq!(image[[20, 50, 50]], 1);
        assert_eq!(image[[80, 50, 50]], 1);
        assert_eq!(image[[50, 40, 50]], 1);
        assert_eq!(image[[50, 60, 50]], 1);
        assert_eq!(image[[50, 50, 45]], 1);
        assert_eq!(image[[50, 50, 55]], 1);

        //assert_eq!(image[[50, 50, 54]] , 0);
        assert_eq!(image[[50, 50, 56]], 0);
        assert_eq!(image[[50, 49, 55]], 0);
        assert_eq!(image[[50, 51, 55]], 0);
        assert_eq!(image[[49, 50, 55]], 0);
        assert_eq!(image[[51, 50, 55]], 0);
    }
}
