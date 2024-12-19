use pyo3::prelude::*;

mod drawing_rs;
mod geometry_rs;

#[pymodule]
fn pyellispeed(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "geometry_rs")?;
    geometry_rs::geometry_rs(&module)?;
    m.add_submodule(&module)?;

    let module = PyModule::new(py, "drawing_rs")?;
    drawing_rs::drawing_rs(&module)?;
    m.add_submodule(&module)?;

    Ok(())
}
