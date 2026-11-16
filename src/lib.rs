// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Noncovalent interaction analysis on an affine Cartesian grid.
//!
//! The caller supplies analytic real-space density derivatives. The crate computes
//! NCI, promolecular IGM, and IGMH descriptors in atomic units.

mod analysis;
mod cli;
mod data;
mod expanded;
mod fchk;
mod io;
mod molden;
#[cfg(feature = "python")]
mod python;
mod viewer;
mod wavefunction;
mod wfn;

pub use analysis::{analyze, Analysis, Descriptor, GradientDifference, InteractionPoint, Method};
pub use cli::run_cli;
pub use data::{
    Atom, Dataset, DensityKind, DensityModel, DensitySample, Grid, ProatomSample, Units, Vec3,
};
pub use expanded::{read_expanded, PeriodicCell};
pub use fchk::read_fchk;
pub use io::{read_dataset, write_csv, write_cube, CubeField};
pub use molden::read_molden;
pub use viewer::{write_html, SurfaceDensity, ViewerOptions};
pub use wavefunction::{read_wfx, Wavefunction, WavefunctionFormat};
pub use wfn::read_wfn;

/// Errors opening, validating, or exporting a dataset.
#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid(_) => None,
            Self::Io(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
