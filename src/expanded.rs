// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::data::{add, cross, dot, scale, sub};
use crate::io::Rows;
use crate::wavefunction::{primitive_at, CoreTerm, Primitive};
use crate::{invalid, Atom, DensitySample, Grid, Result, Units, Vec3, Wavefunction};
use std::io::{BufRead, BufReader, Read};

/// Explicit lattice and finite Bloch image summation parameters, in Bohr.
#[derive(Clone, Copy, Debug)]
pub struct PeriodicCell {
    pub lattice: [Vec3; 3],
    pub periodic_axes: [bool; 3],
    pub image_ranges: [i32; 3],
    dual: [Vec3; 3],
}

impl PeriodicCell {
    fn new(lattice: [Vec3; 3], periodic_axes: [bool; 3], image_ranges: [i32; 3]) -> Result<Self> {
        Grid::new([2; 3], [0.0; 3], lattice, Units::Bohr)?;
        let [a, b, c] = lattice;
        let volume = dot(a, cross(b, c));
        if !volume.is_finite() || volume == 0.0 || !periodic_axes.contains(&true) {
            return Err(invalid(
                "periodic lattice must be invertible and have at least one periodic axis",
            ));
        }
        if image_ranges
            .into_iter()
            .zip(periodic_axes)
            .any(|(n, p)| n < 0 || (!p && n != 0))
        {
            return Err(invalid(
                "image ranges must be nonnegative and vanish on open axes",
            ));
        }
        let dual = [cross(b, c), cross(c, a), cross(a, b)].map(|row| scale(row, volume.recip()));
        if !dual.iter().flatten().all(|x| x.is_finite()) {
            return Err(invalid("lattice inverse exceeds the finite range"));
        }
        let count = image_ranges.into_iter().try_fold(1usize, |count, n| {
            count.checked_mul((n as usize).checked_mul(2)?.checked_add(1)?)
        });
        if !count.is_some_and(|count| count <= 1_000_000) {
            return Err(invalid(
                "periodic image cube exceeds one million translations",
            ));
        }
        Ok(Self {
            lattice,
            periodic_axes,
            image_ranges,
            dual,
        })
    }

    pub(crate) fn wrap(&self, point: Vec3) -> Vec3 {
        let mut position = point;
        for axis in 0..3 {
            if self.periodic_axes[axis] {
                let cells = dot(self.dual[axis], point).floor();
                position = sub(position, scale(self.lattice[axis], cells));
            }
        }
        position
    }

    pub(crate) fn grid(&self, shape: [usize; 3]) -> Result<Grid> {
        if shape.iter().any(|&length| length < 2) {
            return Err(invalid(
                "periodic wavefunction grid needs at least two nodes per axis",
            ));
        }
        let steps = std::array::from_fn(|i| scale(self.lattice[i], 1.0 / (shape[i] - 1) as f64));
        Grid::new(shape, [0.0; 3], steps, Units::Bohr)
    }

    pub(crate) fn translations(&self) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
        let [x, y, z] = self.image_ranges;
        (-x..=x).flat_map(move |i| {
            (-y..=y).flat_map(move |j| {
                (-z..=z).map(move |k| {
                    let image = [f64::from(i), f64::from(j), f64::from(k)];
                    let vector = self
                        .lattice
                        .into_iter()
                        .zip(image)
                        .fold([0.0; 3], |vector, (axis, n)| add(vector, scale(axis, n)));
                    (image, vector)
                })
            })
        })
    }
}

type Pair = [f64; 2];

#[derive(Clone, Debug)]
struct State {
    occupation: f64,
    coefficients: Vec<Vec<Pair>>,
}

#[derive(Clone, Debug)]
struct KPoint {
    weight: f64,
    states: Vec<State>,
    translations: Vec<(Vec3, Pair)>,
}

#[derive(Clone, Debug)]
pub(crate) struct Expanded {
    pub(crate) periodic: Option<PeriodicCell>,
    kpoints: Vec<KPoint>,
}

impl Expanded {
    pub(crate) fn electrons(&self) -> f64 {
        self.kpoints
            .iter()
            .map(|k| k.weight * k.states.iter().map(|state| state.occupation).sum::<f64>())
            .sum()
    }

    pub(crate) fn evaluate(
        &self,
        point: Vec3,
        shift: Vec3,
        atoms: &[Atom],
        primitives: &[Primitive],
    ) -> DensitySample {
        let mut density = DensitySample::default();
        let mut basis = Vec::with_capacity(primitives.len());
        for k in &self.kpoints {
            basis.clear();
            for &primitive in primitives {
                let point = sub(point, sub(atoms[primitive.center].position, shift));
                let mut image_sum = [DensitySample::default(); 2];
                for &(vector, phase) in &k.translations {
                    let g = primitive_at(primitive, sub(point, vector));
                    for (component, scale) in image_sum.iter_mut().zip(phase) {
                        accumulate(component, g, scale);
                    }
                }
                basis.push(image_sum);
            }
            for state in &k.states {
                let weight = k.weight * state.occupation;
                if weight == 0.0 {
                    continue;
                }
                // Components of an orbital spinor are orthogonal in spin space.
                for coefficients in &state.coefficients {
                    let mut real = DensitySample::default();
                    let mut imag = DensitySample::default();
                    for (&[re, im], &[pr, pi]) in coefficients.iter().zip(&basis) {
                        accumulate(&mut real, pr, re);
                        accumulate(&mut real, pi, -im);
                        accumulate(&mut imag, pr, im);
                        accumulate(&mut imag, pi, re);
                    }
                    super::wavefunction::accumulate_orbital_density(&mut density, real, weight);
                    super::wavefunction::accumulate_orbital_density(&mut density, imag, weight);
                }
            }
        }
        density
    }
}

pub(crate) fn accumulate(sum: &mut DensitySample, value: DensitySample, coefficient: f64) {
    sum.density += coefficient * value.density;
    for (entry, value) in sum.gradient.iter_mut().zip(value.gradient) {
        *entry += coefficient * value;
    }
    for (entry, value) in sum.hessian.iter_mut().zip(value.hessian) {
        *entry += coefficient * value;
    }
}

/// Read a versioned complex Gaussian primitive interchange file. The format
/// includes explicit k-point weights, periodic images and optional core density.
pub fn read_expanded(input: impl Read) -> Result<Wavefunction> {
    let mut rows = Rows::new(BufReader::new(input));
    if rows.header("NCIPLOT-WAVEFUNCTION", 1)? != ["1"] || rows.header("units", 1)? != ["bohr"] {
        return Err(rows.error("expanded Gaussian wavefunction requires version 1 and Bohr units"));
    }
    let count = count_field(&mut rows, "atoms")?;
    let mut atoms = Vec::new();
    let mut charges = Vec::new();
    for _ in 0..count {
        let fields = rows.row("atom", 5)?;
        atoms.push(Atom {
            atomic_number: rows.number(&fields[0])?,
            position: rows.triple(&fields[1..4])?,
            fragment: 0,
        });
        charges.push(rows.number(&fields[4])?);
    }
    let periodic = read_cell(&mut rows)?;
    let size = count_field(&mut rows, "primitives")?;
    let mut primitives = Vec::new();
    for _ in 0..size {
        let record = rows.row("primitive", 5)?;
        primitives.push(parse_primitive(&rows, &record)?);
    }
    let mut core = Vec::new();
    for _ in 0..count_field(&mut rows, "core")? {
        let fields = rows.row("core density primitive", 6)?;
        core.push(CoreTerm {
            primitive: parse_primitive(&rows, &fields[..5])?,
            coefficient: rows.number(&fields[5])?,
        });
    }
    let mut kpoints = Vec::new();
    for _ in 0..count_field(&mut rows, "kpoints")? {
        kpoints.push(read_kpoint(&mut rows, periodic.as_ref(), size)?);
    }
    if rows.next()?.is_some() {
        return Err(rows.error("extra data after the wavefunction"));
    }
    let total_weight: f64 = kpoints.iter().map(|point| point.weight).sum();
    if (total_weight - 1.0).abs() > 1.0e-10 || (periodic.is_none() && kpoints.len() != 1) {
        return Err(invalid(
            "k-point weights must sum to one and isolated wavefunctions use one Gamma point",
        ));
    }
    let expanded = Expanded { periodic, kpoints };
    if expanded.electrons() <= 0.0 {
        return Err(invalid(
            "complex wavefunction has no occupied orbital density",
        ));
    }
    Wavefunction::new_expanded(atoms, primitives, expanded)?.with_nuclear_data(charges, core)
}

fn count_field<R: BufRead>(rows: &mut Rows<R>, label: &str) -> Result<usize> {
    let value = rows.header(label, 1)?;
    rows.number(&value[0])
}

fn read_cell<R: BufRead>(rows: &mut Rows<R>) -> Result<Option<PeriodicCell>> {
    let axes = rows.header("periodic", 3)?;
    let axes: [u8; 3] = rows.triple(&axes)?;
    if axes.into_iter().any(|axis| axis > 1) {
        return Err(rows.error("periodic axes require binary 0/1 flags"));
    }
    if axes == [0; 3] {
        return Ok(None);
    }
    let mut lattice = [[0.0; 3]; 3];
    for row in &mut lattice {
        let fields = rows.header("lattice", 3)?;
        *row = rows.triple(&fields)?;
    }
    let images = rows.header("images", 3)?;
    Ok(Some(PeriodicCell::new(
        lattice,
        axes.map(|axis| axis == 1),
        rows.triple(&images)?,
    )?))
}

fn parse_primitive<R: BufRead>(rows: &Rows<R>, fields: &[String]) -> Result<Primitive> {
    let center = rows
        .number::<usize>(&fields[0])?
        .checked_sub(1)
        .ok_or_else(|| rows.error("primitive atom indices start at one"))?;
    let powers: [u8; 3] = rows.triple(&fields[1..4])?;
    if powers.iter().map(|&power| u32::from(power)).sum::<u32>() > 12 {
        return Err(rows.error("complex primitive angular momentum exceeds 12"));
    }
    Ok(Primitive {
        center,
        powers,
        exponent: rows.number(&fields[4])?,
    })
}

fn read_kpoint<R: BufRead>(
    rows: &mut Rows<R>,
    cell: Option<&PeriodicCell>,
    primitive_count: usize,
) -> Result<KPoint> {
    let fields = rows.header("k", 5)?;
    let momentum: Vec3 = rows.triple(&fields[..3])?;
    let weight: f64 = rows.number(&fields[3])?;
    let count: usize = rows.number(&fields[4])?;
    let active = cell.map_or([false; 3], |cell| cell.periodic_axes);
    if !(0.0..=1.0).contains(&weight)
        || count == 0
        || momentum
            .into_iter()
            .zip(active)
            .any(|(k, axis)| !(-1.0..=1.0).contains(&k) || (!axis && k != 0.0))
    {
        return Err(rows.error("invalid k-point coordinates, weight or orbitals"));
    }
    let mut states = Vec::new();
    for _ in 0..count {
        let fields = rows.header("orbital", 2)?;
        let occupation: f64 = rows.number(&fields[0])?;
        let spin: usize = rows.number(&fields[1])?;
        let limit = if spin == 1 { 2.0 } else { 1.0 };
        if !(1..=2).contains(&spin) || !(0.0..=limit).contains(&occupation) {
            return Err(
                rows.error("orbitals need a valid occupation and one or two spin components")
            );
        }
        let mut coefficients = Vec::new();
        for _ in 0..spin {
            let mut component = Vec::new();
            for _ in 0..primitive_count {
                let fields = rows.row("complex coefficient", 2)?;
                let pair = [rows.number(&fields[0])?, rows.number(&fields[1])?];
                if !pair.into_iter().all(f64::is_finite) {
                    return Err(rows.error("nonfinite complex orbital coefficient"));
                }
                component.push(pair);
            }
            coefficients.push(component);
        }
        states.push(State {
            occupation,
            coefficients,
        });
    }
    let translations = match cell {
        Some(cell) => cell
            .translations()
            .map(|(image, vector)| {
                let (sin, cos) = (std::f64::consts::TAU * dot(image, momentum)).sin_cos();
                (vector, [cos, sin])
            })
            .collect(),
        None => vec![([0.0; 3], [1.0, 0.0])],
    };
    Ok(KPoint {
        weight,
        states,
        translations,
    })
}
