// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::{invalid, Result};

pub type Vec3 = [f64; 3];

/// The length unit for an entire input. Densities and their derivatives use
/// the corresponding inverse cubic, quartic, and fifth powers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Units {
    Bohr,
    Angstrom,
}

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

impl Units {
    fn angstrom_ratio(self) -> f64 {
        match self {
            Self::Bohr => 1.0,
            Self::Angstrom => ANGSTROM_PER_BOHR,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityKind {
    Molecular,
    Promolecular,
}

/// Electron-density content of imported wavefunctions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityModel {
    Supplied,
    AllElectron,
    ValenceOnly,
    ReconstructedCore,
}

impl std::fmt::Display for DensityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Supplied => "supplied",
            Self::AllElectron => "all-electron",
            Self::ValenceOnly => "valence-only",
            Self::ReconstructedCore => "reconstructed-core",
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Atom {
    pub atomic_number: u8,
    pub position: Vec3,
    pub fragment: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DensitySample {
    pub density: f64,
    pub gradient: Vec3,
    /// Symmetric Cartesian Hessian packed as xx, xy, xz, yy, yz, zz.
    pub hessian: [f64; 6],
}

#[derive(Clone, Copy, Debug)]
pub struct ProatomSample {
    pub density: f64,
    pub gradient: Vec3,
}

/// Affine grid stored in Bohr. Indices increase with the third axis fastest.
#[derive(Clone, Debug)]
pub struct Grid {
    shape: [usize; 3],
    origin: Vec3,
    steps: [Vec3; 3],
    points: usize,
}

impl Grid {
    pub fn new(shape: [usize; 3], origin: Vec3, steps: [Vec3; 3], units: Units) -> Result<Self> {
        let points = shape
            .into_iter()
            .try_fold(1usize, |product, edge| product.checked_mul(edge))
            .filter(|&points| points > 0)
            .ok_or_else(|| invalid("grid shape must be positive and fit in usize"))?;
        let conversion = units.angstrom_ratio().recip();
        let origin = scale(origin, conversion);
        let steps = steps.map(|step| scale(step, conversion));
        let scale_max = steps
            .iter()
            .flatten()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        if !origin.into_iter().all(f64::is_finite) || !scale_max.is_finite() || scale_max == 0.0 {
            return Err(invalid(
                "grid needs a finite origin and finite nonzero steps",
            ));
        }
        let [a, b, c] = steps.map(|step| step.map(|component| component / scale_max));
        let volume = dot(a, cross(b, c));
        if !volume.is_finite() || volume == 0.0 {
            return Err(invalid("grid steps must be linearly independent"));
        }
        for corner in 0..8 {
            let mut position = origin;
            for axis in 0..3 {
                if corner & (1 << axis) != 0 {
                    position = add(position, scale(steps[axis], (shape[axis] - 1) as f64));
                }
                if !position.into_iter().all(f64::is_finite) {
                    return Err(invalid("grid contains positions outside the finite range"));
                }
            }
        }
        Ok(Self {
            shape,
            origin,
            steps,
            points,
        })
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    pub fn steps(&self) -> [Vec3; 3] {
        self.steps
    }

    pub fn point_count(&self) -> usize {
        self.points
    }

    pub fn flat_index(&self, indices: [usize; 3]) -> Option<usize> {
        indices
            .into_iter()
            .zip(self.shape)
            .all(|(index, size)| index < size)
            .then(|| self.linear(indices))
    }

    pub fn position(&self, indices: [usize; 3]) -> Option<Vec3> {
        self.flat_index(indices).map(|_| self.point(indices))
    }

    pub(crate) fn linear(&self, [i, j, k]: [usize; 3]) -> usize {
        (i * self.shape[1] + j) * self.shape[2] + k
    }

    pub(crate) fn point(&self, indices: [usize; 3]) -> Vec3 {
        indices
            .into_iter()
            .zip(self.steps)
            .fold(self.origin, |position, (index, step)| {
                add(position, scale(step, index as f64))
            })
    }
}

/// Geometry and fields measured at each grid node.
///
/// All fields are converted to Bohr on creation. Proatom samples, when
/// supplied, have point-major layout with the atom index changing fastest.
#[derive(Clone, Debug)]
pub struct Dataset {
    grid: Grid,
    atoms: Vec<Atom>,
    density_kind: DensityKind,
    pub(crate) density_model: DensityModel,
    pub(crate) periodic_cell: Option<crate::PeriodicCell>,
    samples: Vec<DensitySample>,
    proatoms: Option<Vec<ProatomSample>>,
}

impl Dataset {
    pub fn new(
        grid: Grid,
        mut atoms: Vec<Atom>,
        density_kind: DensityKind,
        mut samples: Vec<DensitySample>,
        mut proatoms: Option<Vec<ProatomSample>>,
        units: Units,
    ) -> Result<Self> {
        if atoms.is_empty() || samples.len() != grid.point_count() {
            return Err(invalid(
                "dataset needs atoms and one density sample per grid node",
            ));
        }
        if let Some(proatoms) = &proatoms {
            if grid.point_count().checked_mul(atoms.len()) != Some(proatoms.len()) {
                return Err(invalid(
                    "dataset needs one proatom sample per node and atom",
                ));
            }
        }
        let ratio = units.angstrom_ratio();
        let density_scale = ratio.powi(3);
        let gradient_scale = ratio.powi(4);
        let hessian_scale = ratio.powi(5);
        for atom in &mut atoms {
            atom.position = scale(atom.position, ratio.recip());
            if !(1..=118).contains(&atom.atomic_number)
                || !atom.position.into_iter().all(f64::is_finite)
            {
                return Err(invalid(
                    "atom needs atomic number 1 through 118 and a finite position",
                ));
            }
        }
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.density *= density_scale;
            sample.gradient = scale(sample.gradient, gradient_scale);
            sample.hessian = sample.hessian.map(|component| component * hessian_scale);
            if !valid_density(sample.density, sample.gradient)
                || !sample.hessian.into_iter().all(f64::is_finite)
            {
                return Err(invalid(format!(
                    "invalid density or derivatives at node {index}"
                )));
            }
        }
        if let Some(proatoms) = &mut proatoms {
            for (index, sample) in proatoms.iter_mut().enumerate() {
                sample.density *= density_scale;
                sample.gradient = scale(sample.gradient, gradient_scale);
                if !valid_density(sample.density, sample.gradient) {
                    return Err(invalid(format!(
                        "invalid proatom density or gradient at node {} atom {}",
                        index / atoms.len(),
                        index % atoms.len()
                    )));
                }
            }
        }
        Ok(Self {
            grid,
            atoms,
            density_kind,
            density_model: DensityModel::Supplied,
            periodic_cell: None,
            samples,
            proatoms,
        })
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    pub fn density_kind(&self) -> DensityKind {
        self.density_kind
    }

    pub fn density_model(&self) -> DensityModel {
        self.density_model
    }

    pub fn periodic_cell(&self) -> Option<&crate::PeriodicCell> {
        self.periodic_cell.as_ref()
    }

    pub fn density_samples(&self) -> &[DensitySample] {
        &self.samples
    }

    pub fn proatom_samples(&self) -> Option<&[ProatomSample]> {
        self.proatoms.as_deref()
    }
}

fn valid_density(density: f64, gradient: Vec3) -> bool {
    density >= 0.0 && density.is_finite() && gradient.into_iter().all(f64::is_finite)
}

pub(crate) fn add(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|axis| a[axis] + b[axis])
}

pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

pub(crate) fn scale(vector: Vec3, factor: f64) -> Vec3 {
    vector.map(|value| value * factor)
}

pub(crate) fn norm([x, y, z]: Vec3) -> f64 {
    x.hypot(y).hypot(z)
}

pub(crate) fn dot(a: Vec3, b: Vec3) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|axis| {
        let j = (axis + 1) % 3;
        let k = (axis + 2) % 3;
        a[j] * b[k] - a[k] * b[j]
    })
}

pub(crate) const ELEMENTS: [&str; 119] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];
pub(crate) fn atomic_number(symbol: &str) -> Option<u8> {
    ELEMENTS
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, element)| element.eq_ignore_ascii_case(symbol))
        .map(|(index, _)| index as u8)
}
