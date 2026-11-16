// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::{
    analyze, read_dataset, run_cli, write_csv, write_cube, write_html, Analysis, CubeField,
    Dataset, Descriptor, Error, Grid, Method, SurfaceDensity, Units, Vec3, ViewerOptions,
    Wavefunction, WavefunctionFormat,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyDict},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

fn python_error(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        Error::Io(error) => error.into(),
    }
}

fn wavefunction_input(path: &Path, format: Option<&str>) -> crate::Result<Wavefunction> {
    let format = if let Some(format) = format {
        format.parse()?
    } else {
        WavefunctionFormat::detect(path).ok_or_else(|| {
            crate::invalid(
                "cannot determine wavefunction format from path; supply format explicitly",
            )
        })?
    };
    format.read(BufReader::new(File::open(path)?))
}

#[pyclass(name = "Grid", module = "nciplot_rs", frozen)]
struct PyGrid(Grid);

#[pymethods]
impl PyGrid {
    /// An affine Cartesian grid. Coordinates and steps use the given length unit.
    #[new]
    #[pyo3(signature = (shape, origin, steps, *, units = "bohr"))]
    fn new(shape: [usize; 3], origin: Vec3, steps: [Vec3; 3], units: &str) -> PyResult<Self> {
        let units = match units {
            "bohr" => Units::Bohr,
            "angstrom" => Units::Angstrom,
            _ => return Err(PyValueError::new_err("units must be bohr or angstrom")),
        };
        Ok(Self(
            Grid::new(shape, origin, steps, units).map_err(python_error)?,
        ))
    }

    #[getter]
    fn shape(&self) -> [usize; 3] {
        self.0.shape()
    }

    #[getter]
    fn origin(&self) -> Vec3 {
        self.0.origin()
    }

    #[getter]
    fn steps(&self) -> [Vec3; 3] {
        self.0.steps()
    }
}

#[pyclass(name = "Wavefunction", module = "nciplot_rs", frozen)]
struct PyWavefunction(Wavefunction);

#[pymethods]
impl PyWavefunction {
    #[getter]
    fn electron_count(&self) -> f64 {
        self.0.electron_count()
    }

    #[getter]
    fn density_model(&self) -> String {
        self.0.density_model().to_string()
    }

    #[getter]
    fn periodic_cell<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.0
            .periodic_cell()
            .map(|cell| cell_report(py, cell))
            .transpose()
    }

    /// Atomic numbers and Cartesian positions in Bohr in input order.
    #[getter]
    fn atoms(&self) -> Vec<(u8, Vec3)> {
        self.0
            .atoms()
            .iter()
            .map(|atom| (atom.atomic_number, atom.position))
            .collect()
    }

    /// Density, gradient and packed Hessian in atomic units at a Bohr point.
    fn density_at(&self, point: Vec3) -> PyResult<(f64, Vec3, [f64; 6])> {
        let fields = self.0.density_at(point).map_err(python_error)?;
        Ok((fields.density, fields.gradient, fields.hessian))
    }

    /// Sample molecular density and optional neutral-atom references.
    #[pyo3(signature = (*, shape = [41; 3], padding = 5.0, fragments = None, proatoms = None, grid = None))]
    fn sample(
        &self,
        py: Python<'_>,
        shape: [usize; 3],
        padding: f64,
        fragments: Option<Vec<usize>>,
        proatoms: Option<BTreeMap<u8, PathBuf>>,
        grid: Option<&PyGrid>,
    ) -> PyResult<PyDataset> {
        let grid = match grid {
            Some(grid) => grid.0.clone(),
            None => self.0.bounding_grid(shape, padding).map_err(python_error)?,
        };
        let wavefunction = &self.0;
        let dataset = py
            .allow_threads(move || -> crate::Result<Dataset> {
                let references = proatoms
                    .map(|paths| {
                        paths
                            .into_iter()
                            .map(|(element, path)| Ok((element, wavefunction_input(&path, None)?)))
                            .collect::<crate::Result<BTreeMap<_, _>>>()
                    })
                    .transpose()?;
                wavefunction.sample_grid(grid, fragments.as_deref(), references.as_ref())
            })
            .map_err(python_error)?;
        Ok(PyDataset(Arc::new(dataset)))
    }
}

#[pyfunction]
#[pyo3(signature = (path, *, format = None))]
fn load_wavefunction(
    py: Python<'_>,
    path: PathBuf,
    format: Option<&str>,
) -> PyResult<PyWavefunction> {
    py.allow_threads(|| wavefunction_input(&path, format))
        .map(PyWavefunction)
        .map_err(python_error)
}

#[pyclass(name = "Dataset", module = "nciplot_rs", frozen)]
struct PyDataset(Arc<Dataset>);

#[pymethods]
impl PyDataset {
    #[getter]
    fn grid(&self) -> PyGrid {
        PyGrid(self.0.grid().clone())
    }

    /// Evaluate NCI, IGM or either IGMH convention on the grid.
    #[pyo3(signature = (method = "nci"))]
    fn analyze(&self, py: Python<'_>, method: &str) -> PyResult<PyAnalysis> {
        let method = method.parse::<Method>().map_err(python_error)?;
        let dataset = Arc::clone(&self.0);
        let analysis = py
            .allow_threads(|| analyze(&dataset, method))
            .map_err(python_error)?;
        Ok(PyAnalysis { dataset, analysis })
    }
}

#[pyfunction]
fn load_dataset(py: Python<'_>, path: PathBuf) -> PyResult<PyDataset> {
    py.allow_threads(|| read_dataset(BufReader::new(File::open(path)?)))
        .map(|dataset| PyDataset(Arc::new(dataset)))
        .map_err(python_error)
}

#[pyclass(name = "ViewerOptions", module = "nciplot_rs", frozen)]
struct PyViewerOptions {
    field: Option<Descriptor>,
    isovalue: Option<f64>,
    density: SurfaceDensity,
    density_range: Option<(Option<f64>, Option<f64>)>,
    color_range: Option<(f64, f64)>,
    scatter_limit: Option<f64>,
}

#[pymethods]
impl PyViewerOptions {
    /// Initial settings for the single offline HTML viewer.
    #[new]
    #[pyo3(signature = (*, field = None, isovalue = None, density = "rho", density_range = None,
        color_range = None, scatter_limit = None))]
    fn new(
        field: Option<&str>,
        isovalue: Option<f64>,
        density: &str,
        density_range: Option<(Option<f64>, Option<f64>)>,
        color_range: Option<(f64, f64)>,
        scatter_limit: Option<f64>,
    ) -> PyResult<Self> {
        let density = match density {
            "rho" => SurfaceDensity::Unsigned,
            "signed" => SurfaceDensity::Signed,
            _ => return Err(PyValueError::new_err("density must be rho or signed")),
        };
        Ok(Self {
            field: field
                .map(str::parse::<Descriptor>)
                .transpose()
                .map_err(python_error)?,
            isovalue,
            density,
            density_range,
            color_range,
            scatter_limit,
        })
    }
}

impl PyViewerOptions {
    fn to_rust(&self, field: Descriptor) -> ViewerOptions {
        let mut options = ViewerOptions::for_density(self.field.unwrap_or(field), self.density);
        options.isovalue = self.isovalue.unwrap_or(options.isovalue);
        if let Some((minimum, maximum)) = self.density_range {
            options.surface_density_min = minimum.unwrap_or(options.surface_density_min);
            options.surface_density_max = maximum.unwrap_or(options.surface_density_max);
        }
        if let Some((minimum, maximum)) = self.color_range {
            options.color_min = minimum;
            options.color_max = maximum;
        }
        options.scatter_density_limit = self.scatter_limit.unwrap_or(options.scatter_density_limit);
        options
    }
}

#[pyclass(name = "Analysis", module = "nciplot_rs", frozen)]
struct PyAnalysis {
    dataset: Arc<Dataset>,
    analysis: Analysis,
}

#[pymethods]
impl PyAnalysis {
    #[getter]
    fn method(&self) -> String {
        self.analysis.method.to_string()
    }

    /// Return grid metadata, scalar fields, Cartesian derivatives and closure diagnostics.
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = PyDict::new(py);
        let grid = self.dataset.grid();
        let points = &self.analysis.points;
        let samples = self.dataset.density_samples();
        report.set_item("method", self.method())?;
        report.set_item("density_model", self.dataset.density_model().to_string())?;
        report.set_item(
            "periodic_cell",
            self.dataset
                .periodic_cell()
                .map(|cell| cell_report(py, cell))
                .transpose()?,
        )?;
        report.set_item("shape", grid.shape())?;
        report.set_item("origin", grid.origin())?;
        report.set_item("steps", grid.steps())?;
        report.set_item(
            "density",
            points.iter().map(|point| point.density).collect::<Vec<_>>(),
        )?;
        report.set_item(
            "gradient",
            samples
                .iter()
                .map(|sample| sample.gradient)
                .collect::<Vec<_>>(),
        )?;
        report.set_item(
            "hessian",
            samples
                .iter()
                .map(|sample| sample.hessian)
                .collect::<Vec<_>>(),
        )?;
        report.set_item(
            "lambda2",
            points.iter().map(|point| point.lambda2).collect::<Vec<_>>(),
        )?;
        report.set_item(
            "signed_density",
            points
                .iter()
                .map(|point| point.signed_density)
                .collect::<Vec<_>>(),
        )?;
        for (key, field) in [
            ("rdg", Descriptor::Rdg),
            ("delta_g", Descriptor::Total),
            ("delta_g_inter", Descriptor::Inter),
            ("delta_g_intra", Descriptor::Intra),
        ] {
            report.set_item(
                key,
                points
                    .iter()
                    .map(|point| point.value(field))
                    .collect::<Vec<_>>(),
            )?;
        }
        report.set_item("zero_density_points", self.analysis.zero_density_points)?;
        report.set_item("density_residual", self.analysis.density_residual)?;
        report.set_item("gradient_residual", self.analysis.gradient_residual)?;
        let units = PyDict::new(py);
        for (key, value) in [
            ("origin", "bohr"),
            ("steps", "bohr"),
            ("density", "e/bohr^3"),
            ("gradient", "e/bohr^4"),
            ("hessian", "e/bohr^5"),
            ("lambda2", "e/bohr^5"),
            ("signed_density", "e/bohr^3"),
            ("rdg", "1"),
            ("delta_g", "e/bohr^4"),
            ("delta_g_inter", "e/bohr^4"),
            ("delta_g_intra", "e/bohr^4"),
            ("density_residual", "e/bohr^3"),
            ("gradient_residual", "e/bohr^4"),
        ] {
            units.set_item(key, value)?;
        }
        report.set_item("units", units)?;
        Ok(report)
    }

    fn write_csv(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.allow_threads(|| save(&path, |out| write_csv(out, &self.dataset, &self.analysis)))
            .map_err(python_error)
    }

    /// Write density, signed-density, or a selected descriptor as a CUBE file.
    #[pyo3(signature = (path, field = None))]
    fn write_cube(&self, py: Python<'_>, path: PathBuf, field: Option<&str>) -> PyResult<()> {
        let field = match field {
            None => CubeField::Descriptor(self.analysis.method.default_descriptor(&self.dataset)),
            Some("density") => CubeField::Density,
            Some("signed-density") => CubeField::SignedDensity,
            Some(field) => CubeField::Descriptor(field.parse().map_err(python_error)?),
        };
        py.allow_threads(|| {
            save(&path, |out| {
                write_cube(out, &self.dataset, &self.analysis, field)
            })
        })
        .map_err(python_error)
    }

    #[pyo3(signature = (path, *, options = None))]
    fn write_html(
        &self,
        py: Python<'_>,
        path: PathBuf,
        options: Option<&PyViewerOptions>,
    ) -> PyResult<()> {
        let field = self.analysis.method.default_descriptor(&self.dataset);
        let options = options.map_or_else(
            || ViewerOptions::new(field),
            |options| options.to_rust(field),
        );
        py.allow_threads(|| {
            save(&path, |out| {
                write_html(out, &self.dataset, &self.analysis, options)
            })
        })
        .map_err(python_error)
    }
}

fn save(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> crate::Result<()>,
) -> crate::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    write(&mut out)?;
    out.flush()?;
    Ok(())
}

fn cell_report<'py>(py: Python<'py>, cell: &crate::PeriodicCell) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("lattice_bohr", cell.lattice)?;
    result.set_item("periodic_axes", cell.periodic_axes)?;
    result.set_item("image_ranges", cell.image_ranges)?;
    Ok(result)
}

#[pyfunction]
fn _cli<'py>(
    py: Python<'py>,
    args: Vec<OsString>,
) -> PyResult<(u8, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let (status, stdout, stderr) = py.allow_threads(|| -> std::io::Result<_> {
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let status = run_cli(args, &mut stdout, &mut stderr)?;
        Ok((status, stdout, stderr))
    })?;
    Ok((status, PyBytes::new(py, &stdout), PyBytes::new(py, &stderr)))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<PyGrid>()?;
    module.add_class::<PyWavefunction>()?;
    module.add_class::<PyDataset>()?;
    module.add_class::<PyAnalysis>()?;
    module.add_class::<PyViewerOptions>()?;
    module.add_function(wrap_pyfunction!(load_wavefunction, module)?)?;
    module.add_function(wrap_pyfunction!(load_dataset, module)?)?;
    module.add_function(wrap_pyfunction!(_cli, module)?)?;
    Ok(())
}
