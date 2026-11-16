// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::{
    invalid, Analysis, Atom, Dataset, DensityKind, DensitySample, Descriptor, Grid, Method,
    ProatomSample, Result, Units,
};
use std::io::{BufRead, Write};
use std::str::FromStr;

pub(crate) struct Rows<R> {
    lines: std::iter::Enumerate<std::io::Lines<R>>,
    line: usize,
}

impl<R: BufRead> Rows<R> {
    pub(crate) fn new(input: R) -> Self {
        Self {
            lines: input.lines().enumerate(),
            line: 0,
        }
    }

    pub(crate) fn next(&mut self) -> Result<Option<Vec<String>>> {
        for (index, line) in self.lines.by_ref() {
            self.line = index + 1;
            let line = line?;
            let data = line.split('#').next().unwrap_or("").trim();
            if !data.is_empty() {
                return Ok(Some(data.split_whitespace().map(str::to_owned).collect()));
            }
        }
        Ok(None)
    }

    pub(crate) fn row(&mut self, name: &str, fields: usize) -> Result<Vec<String>> {
        let tokens = self
            .next()?
            .ok_or_else(|| invalid(format!("missing {name} after line {}", self.line)))?;
        if tokens.len() != fields {
            return Err(self.error(format!("expected {fields} fields for {name}")));
        }
        Ok(tokens)
    }

    pub(crate) fn header(&mut self, name: &str, fields: usize) -> Result<Vec<String>> {
        let mut tokens = self.row(name, fields + 1)?;
        if tokens.remove(0) != name {
            return Err(self.error(format!("expected {name}")));
        }
        Ok(tokens)
    }

    pub(crate) fn number<T: FromStr>(&self, token: &str) -> Result<T> {
        token
            .parse()
            .map_err(|_| self.error(format!("invalid number {token}")))
    }

    pub(crate) fn triple<T: FromStr>(&self, tokens: &[String]) -> Result<[T; 3]> {
        Ok([
            self.number(&tokens[0])?,
            self.number(&tokens[1])?,
            self.number(&tokens[2])?,
        ])
    }

    pub(crate) fn error(&self, message: impl std::fmt::Display) -> crate::Error {
        invalid(format!("line {}: {message}", self.line))
    }
}

fn read_grid<R: BufRead>(rows: &mut Rows<R>, units: Units) -> Result<Grid> {
    let shape = rows.header("shape", 3)?;
    let origin = rows.header("origin", 3)?;
    let mut steps = [[0.0; 3]; 3];
    for step in &mut steps {
        let values = rows.header("step", 3)?;
        *step = rows.triple(&values)?;
    }
    Grid::new(rows.triple(&shape)?, rows.triple(&origin)?, steps, units)
}

/// Read the versioned analytic density sample interchange format.
pub fn read_dataset(input: impl BufRead) -> Result<Dataset> {
    let mut rows = Rows::new(input);
    if rows.header("NCIPLOT-RS", 1)? != ["1"] {
        return Err(rows.error("unsupported dataset version"));
    }
    let units = match rows.header("units", 1)?[0].as_str() {
        "bohr" => Units::Bohr,
        "angstrom" => Units::Angstrom,
        _ => return Err(rows.error("length unit must be bohr or angstrom")),
    };
    let grid = read_grid(&mut rows, units)?;
    let count = rows.header("atoms", 1)?;
    let count: usize = rows.number(&count[0])?;
    if count == 0 {
        return Err(rows.error("at least one atom is required"));
    }
    let mut atoms = Vec::new();
    for _ in 0..count {
        let fields = rows.row("atom", 5)?;
        atoms.push(Atom {
            atomic_number: rows.number(&fields[0])?,
            position: rows.triple(&fields[1..4])?,
            fragment: rows.number(&fields[4])?,
        });
    }
    let sample_fields = rows.header("samples", 2)?;
    let kind = match sample_fields[0].as_str() {
        "molecular" => DensityKind::Molecular,
        "promolecular" => DensityKind::Promolecular,
        _ => return Err(rows.error("density must be molecular or promolecular")),
    };
    let mut proatoms = match sample_fields[1].as_str() {
        "density-only" => None,
        "with-proatoms" => Some(Vec::new()),
        _ => return Err(rows.error("samples must be density-only or with-proatoms")),
    };
    let mut samples = Vec::new();
    for _ in 0..grid.point_count() {
        let fields = rows.row("density sample", 10)?;
        let values = fields
            .iter()
            .map(|value| rows.number(value))
            .collect::<Result<Vec<f64>>>()?;
        samples.push(DensitySample {
            density: values[0],
            gradient: values[1..4].try_into().expect("three gradient components"),
            hessian: values[4..10].try_into().expect("six Hessian components"),
        });
        if let Some(proatoms) = &mut proatoms {
            for _ in 0..atoms.len() {
                let fields = rows.row("neutral proatom sample", 4)?;
                proatoms.push(ProatomSample {
                    density: rows.number(&fields[0])?,
                    gradient: rows.triple(&fields[1..4])?,
                });
            }
        }
    }
    if rows.next()?.is_some() {
        return Err(rows.error("unexpected data after the final grid node"));
    }
    Dataset::new(grid, atoms, kind, samples, proatoms, units)
}

/// Write all scalar fields and Cartesian coordinates as CSV. Empty cells mark
/// an undefined reduced gradient or the absence of an IGM/IGMH field.
pub fn write_csv(mut out: impl Write, dataset: &Dataset, analysis: &Analysis) -> Result<()> {
    validate_analysis(dataset, analysis)?;
    writeln!(
        out,
        "x_bohr,y_bohr,z_bohr,density_e_bohr3,lambda2_e_bohr5,signed_density_e_bohr3,grad_x_e_bohr4,grad_y_e_bohr4,grad_z_e_bohr4,hess_xx_e_bohr5,hess_xy_e_bohr5,hess_xz_e_bohr5,hess_yy_e_bohr5,hess_yz_e_bohr5,hess_zz_e_bohr5,rdg,delta_g_e_bohr4,delta_g_inter_e_bohr4,delta_g_intra_e_bohr4"
    )?;
    let [nx, ny, nz] = dataset.grid().shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let indices = [i, j, k];
                let [x, y, z] = dataset.grid().point(indices);
                let index = dataset.grid().linear(indices);
                let point = analysis.points[index];
                write!(
                    out,
                    "{x:.16e},{y:.16e},{z:.16e},{:.16e},{:.16e},{:.16e}",
                    point.density, point.lambda2, point.signed_density
                )?;
                let sample = dataset.density_samples()[index];
                for value in sample.gradient.into_iter().chain(sample.hessian) {
                    write!(out, ",{value:.16e}")?;
                }
                for descriptor in [
                    Descriptor::Rdg,
                    Descriptor::Total,
                    Descriptor::Inter,
                    Descriptor::Intra,
                ] {
                    write!(out, ",")?;
                    if let Some(value) = point.value(descriptor) {
                        write!(out, "{value:.16e}")?;
                    }
                }
                writeln!(out)?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum CubeField {
    Density,
    SignedDensity,
    Descriptor(Descriptor),
}

/// Write an atomic-unit scalar CUBE volume. An undefined reduced gradient is
/// written as zero because CUBE has no missing-value representation.
pub fn write_cube(
    mut out: impl Write,
    dataset: &Dataset,
    analysis: &Analysis,
    field: CubeField,
) -> Result<()> {
    validate_analysis(dataset, analysis)?;
    let description = match field {
        CubeField::Density => "rho (e/Bohr^3)".to_owned(),
        CubeField::SignedDensity => "sign(lambda2) rho (e/Bohr^3)".to_owned(),
        CubeField::Descriptor(descriptor) => {
            validate_descriptor(analysis, descriptor)?;
            if descriptor == Descriptor::Rdg {
                "reduced density gradient (dimensionless, vacuum = 0)".to_owned()
            } else {
                format!("delta g {descriptor} (e/Bohr^4)")
            }
        }
    };
    writeln!(out, "nciplot-rs {}: {description}", analysis.method)?;
    writeln!(out, "Atomic units, third grid axis fastest")?;
    let [x, y, z] = dataset.grid().origin();
    writeln!(out, "{} {x:.12e} {y:.12e} {z:.12e}", dataset.atoms().len())?;
    for (count, [x, y, z]) in dataset
        .grid()
        .shape()
        .into_iter()
        .zip(dataset.grid().steps())
    {
        writeln!(out, "{count} {x:.12e} {y:.12e} {z:.12e}")?;
    }
    for atom in dataset.atoms() {
        let [x, y, z] = atom.position;
        writeln!(
            out,
            "{} {:.7} {x:.12e} {y:.12e} {z:.12e}",
            atom.atomic_number,
            f64::from(atom.atomic_number)
        )?;
    }
    for row in analysis.points.chunks(dataset.grid().shape()[2]) {
        for values in row.chunks(6) {
            for point in values {
                let value = match field {
                    CubeField::Density => point.density,
                    CubeField::SignedDensity => point.signed_density,
                    CubeField::Descriptor(descriptor) => point.value(descriptor).unwrap_or(0.0),
                };
                write!(out, " {value:.12e}")?;
            }
            writeln!(out)?;
        }
    }
    Ok(())
}

pub(crate) fn validate_analysis(dataset: &Dataset, analysis: &Analysis) -> Result<()> {
    if dataset.grid().point_count() != analysis.points.len() {
        return Err(invalid("analysis point count does not match the grid"));
    }
    Ok(())
}

pub(crate) fn validate_descriptor(analysis: &Analysis, descriptor: Descriptor) -> Result<()> {
    if descriptor != Descriptor::Rdg && analysis.method == Method::Nci {
        return Err(invalid(
            "an NCI analysis only supplies the reduced density gradient descriptor",
        ));
    }
    Ok(())
}
