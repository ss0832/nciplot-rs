// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::wavefunction::{numbers, Orbital, Primitive};
use crate::{invalid, Atom, Result, Wavefunction};
use std::{collections::BTreeMap, io::Read};

struct Section<'a> {
    name: String,
    qualifier: &'a str,
    lines: Vec<&'a str>,
}

/// Read real all-electron Molden orbitals. The format's normalized Gaussian
/// contractions are expanded into unnormalized Cartesian primitives.
pub fn read_molden(mut input: impl Read) -> Result<Wavefunction> {
    let mut document = String::new();
    input.read_to_string(&mut document)?;
    let mut sections: Vec<Section<'_>> = Vec::new();
    for line in document.lines() {
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, qualifier)) = line
            .strip_prefix('[')
            .and_then(|header| header.split_once(']'))
        {
            sections.push(Section {
                name: name.to_ascii_uppercase(),
                qualifier: qualifier.trim(),
                lines: Vec::new(),
            });
        } else {
            let section = sections
                .last_mut()
                .ok_or_else(|| invalid("missing Molden format header"))?;
            section.lines.push(line);
        }
    }
    if sections.first().map(|section| section.name.as_str()) != Some("MOLDEN FORMAT") {
        return Err(invalid("missing Molden format header"));
    }
    if sections.iter().any(|section| {
        matches!(
            section.name.as_str(),
            "ECP" | "PSEUDO" | "STO" | "CELL" | "LATTICE" | "SPINOR" | "MO COMPLEX"
        )
    }) {
        return Err(invalid(
            "Molden input requires real all-electron Gaussian orbitals of an isolated molecule",
        ));
    }
    let has = |name: &str| sections.iter().any(|section| section.name == name);
    let spherical_df = has("5D") || has("5D7F");
    let spherical_d = spherical_df || has("5D10F");
    let spherical_f = spherical_df || has("7F");
    if (spherical_d && has("6D")) || (spherical_f && has("5D10F")) || (has("9G") && has("15G")) {
        return Err(invalid(
            "conflicting Molden Cartesian and spherical shell tags",
        ));
    }
    let spherical = [
        false,
        false,
        spherical_d,
        spherical_f && !has("10F"),
        has("9G"),
    ];
    let atom_section = unique_section(&sections, "ATOMS")?;
    let unit = atom_section
        .qualifier
        .trim_matches(['(', ')'])
        .to_ascii_uppercase();
    let scale = match unit.as_str() {
        "AU" => 1.0,
        "ANGS" => 0.529_177_210_903_f64.recip(),
        _ => return Err(invalid("Molden [Atoms] requires AU or Angs coordinates")),
    };
    let mut atoms = Vec::new();
    for (index, line) in atom_section.lines.iter().enumerate() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 6 || fields[1].parse::<usize>().ok() != Some(index + 1) {
            return Err(invalid("Molden [Atoms] requires consecutive atom indices"));
        }
        let axial = numbers::<f64>(&fields[3..].join(" "))?;
        atoms.push(Atom {
            atomic_number: fields[2]
                .parse()
                .map_err(|_| invalid("invalid Molden atomic number"))?,
            position: [axial[0] * scale, axial[1] * scale, axial[2] * scale],
            fragment: 0,
        });
    }
    let mut basis = Basis::default();
    basis.read_molden(unique_section(&sections, "GTO")?, atoms.len(), spherical)?;
    let orbitals = basis.read_orbitals(unique_section(&sections, "MO")?)?;
    Wavefunction::new(atoms, basis.primitives, orbitals)
}

fn unique_section<'a>(sections: &'a [Section], name: &str) -> Result<&'a Section<'a>> {
    let mut matches = sections.iter().filter(|section| section.name == name);
    let section = matches
        .next()
        .ok_or_else(|| invalid(format!("missing Molden [{name}] section")))?;
    if matches.next().is_some() {
        return Err(invalid(format!("duplicate Molden [{name}] section")));
    }
    Ok(section)
}

#[derive(Default)]
pub(crate) struct Basis {
    pub(crate) primitives: Vec<Primitive>,
    pub(crate) functions: Vec<Vec<(usize, f64)>>,
    indices: BTreeMap<(usize, u64, [u8; 3]), usize>,
}

impl Basis {
    fn read_molden(
        &mut self,
        section: &Section,
        atom_count: usize,
        spherical: [bool; 5],
    ) -> Result<()> {
        let mut center = None;
        let mut lines = section.lines.iter();
        while let Some(&line) = lines.next() {
            // Each atom index occurs in exactly one ordered basis block.
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() == 2 && fields[1] == "0" {
                let index = fields[0]
                    .parse::<usize>()
                    .map_err(|_| invalid("invalid Molden GTO center"))?;
                let index = index
                    .checked_sub(1)
                    .filter(|&index| index < atom_count)
                    .ok_or_else(|| invalid("Molden GTO center is outside the atom list"))?;
                if index != center.map_or(0, |previous| previous + 1) {
                    return Err(invalid(
                        "Molden GTO center blocks must be consecutive and ordered",
                    ));
                }
                center = Some(index);
                continue;
            }
            let center =
                center.ok_or_else(|| invalid("Molden shell lacks a preceding GTO center"))?;
            if fields.len() != 3 || numbers::<f64>(fields[2])? != [1.0] {
                return Err(invalid(
                    "Molden GTO shell needs a primitive count and unit scale",
                ));
            }
            let label = fields[0].to_ascii_lowercase();
            let count = fields[1]
                .parse::<usize>()
                .ok()
                .filter(|&count| count > 0)
                .ok_or_else(|| invalid("invalid Molden primitive count"))?;
            let degrees: &[u8] = match label.as_str() {
                "s" => &[0],
                "p" => &[1],
                "sp" => &[0, 1],
                "d" => &[2],
                "f" => &[3],
                "g" => &[4],
                _ => return Err(invalid(format!("unsupported Molden shell {label}"))),
            };
            let mut parameters = Vec::new();
            for _ in 0..count {
                let line = lines
                    .next()
                    .ok_or_else(|| invalid("Molden shell has too few primitives"))?;
                let values: Vec<f64> = numbers(line)?;
                if values.len() != degrees.len() + 1
                    || values[0] <= 0.0
                    || values.iter().any(|v| !v.is_finite())
                {
                    return Err(invalid(
                        "invalid Molden primitive exponent or contraction coefficients",
                    ));
                }
                parameters.push(values);
            }
            for (column, &l) in degrees.iter().enumerate() {
                self.add_shell(
                    center,
                    l,
                    spherical[usize::from(l)],
                    column + 1,
                    &parameters,
                )?;
            }
        }
        if center.map(|index| index + 1) != Some(atom_count) || self.functions.is_empty() {
            return Err(invalid("Molden GTO basis did not cover every atom"));
        }
        Ok(())
    }

    pub(crate) fn add_shell(
        &mut self,
        center: usize,
        l: u8,
        spherical: bool,
        column: usize,
        parameters: &[Vec<f64>],
    ) -> Result<()> {
        let degree = f64::from(l) + 1.5;
        // Overlap of normalized equal-angular-momentum primitives at one center.
        let contraction_norm = parameters
            .iter()
            .flat_map(|a| {
                parameters.iter().map(move |b| {
                    let (alpha, beta) = (a[0], b[0]);
                    let ratio = alpha.min(beta) / alpha.max(beta);
                    a[column] * b[column] * (2.0 * ratio.sqrt() / (1.0 + ratio)).powf(degree)
                })
            })
            .sum::<f64>()
            .sqrt();
        if !contraction_norm.is_finite() || contraction_norm <= 0.0 {
            return Err(invalid("Gaussian contraction has a zero or invalid norm"));
        }
        for polynomial in angular_polynomials(l, spherical) {
            let angular_norm = polynomial
                .iter()
                .flat_map(|(a, ca)| {
                    polynomial.iter().map(move |(b, cb)| {
                        ca * cb
                            * (0..3)
                                .map(|axis| gaussian_moment(a[axis] + b[axis]))
                                .product::<f64>()
                    })
                })
                .sum::<f64>()
                .sqrt();
            let mut expansion = Vec::new();
            for values in parameters {
                let exponent = values[0];
                let common = values[column] * (2.0 * exponent).powf(degree / 2.0)
                    / (contraction_norm * angular_norm);
                for &(powers, coefficient) in &polynomial {
                    let index = *self
                        .indices
                        .entry((center, exponent.to_bits(), powers))
                        .or_insert_with(|| {
                            let index = self.primitives.len();
                            self.primitives.push(Primitive {
                                center,
                                exponent,
                                powers,
                            });
                            index
                        });
                    expansion.push((index, common * coefficient));
                }
            }
            self.functions.push(expansion);
        }
        Ok(())
    }

    pub(crate) fn function_count(&self) -> usize {
        self.functions.len()
    }

    pub(crate) fn expand_orbital(
        &self,
        occupancy: f64,
        weights: impl Iterator<Item = f64>,
    ) -> Orbital {
        let mut coefficients = vec![0.0; self.primitives.len()];
        for (weight, primitive) in weights.zip(&self.functions) {
            if weight != 0.0 {
                for &(index, expansion) in primitive {
                    coefficients[index] += weight * expansion;
                }
            }
        }
        Orbital {
            occupancy,
            coefficients,
        }
    }

    fn read_orbitals(&self, section: &Section) -> Result<Vec<Orbital>> {
        let mut orbitals = Vec::new();
        let mut metadata = Metadata::default();
        let mut coeffs = vec![None; self.functions.len()];
        for &line in &section.lines {
            if let Some((key, value)) = line.split_once('=') {
                if coeffs.iter().any(Option::is_some) {
                    self.push_orbital(&mut orbitals, &mut metadata, &mut coeffs)?;
                }
                let key = key.trim().to_ascii_uppercase();
                let value = value.trim();
                match key.as_str() {
                    "OCCUP" if metadata.occupancy.is_none() => {
                        let occupations: Vec<f64> = numbers(value)?;
                        if let [occupancy] = occupations.as_slice() {
                            metadata.occupancy = Some(*occupancy);
                        } else {
                            return Err(invalid("Molden orbital needs one occupation number"));
                        }
                    }
                    "SPIN"
                        if !metadata.spin_seen
                            && matches!(value.to_ascii_uppercase().as_str(), "ALPHA" | "BETA") =>
                    {
                        metadata.spin_seen = true
                    }
                    "SYM" if !metadata.symmetry_seen && !value.is_empty() => {
                        metadata.symmetry_seen = true
                    }
                    "ENE" if !metadata.energy_seen => {
                        let energies: Vec<f64> = numbers(value)?;
                        if energies.len() != 1 || !energies[0].is_finite() {
                            return Err(invalid("Molden orbital needs one finite energy"));
                        }
                        metadata.energy_seen = true;
                    }
                    _ => {
                        return Err(invalid(format!(
                            "unsupported or repeated Molden orbital field {key}"
                        )))
                    }
                }
            } else {
                let fields: Vec<_> = line.split_whitespace().collect();
                if fields.len() != 2 {
                    return Err(invalid(
                        "Molden orbital coefficients must have a basis index and one real value",
                    ));
                }
                let index = fields[0]
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| index.checked_sub(1))
                    .and_then(|index| coeffs.get_mut(index))
                    .ok_or_else(|| invalid("invalid Molden orbital basis index"))?;
                if index.is_some() {
                    return Err(invalid("duplicate Molden orbital basis index"));
                }
                let values: Vec<f64> = numbers(fields[1])?;
                if !values[0].is_finite() {
                    return Err(invalid("Molden orbital coefficient must be finite"));
                }
                *index = Some(values[0]);
            }
        }
        self.push_orbital(&mut orbitals, &mut metadata, &mut coeffs)?;
        Ok(orbitals)
    }

    fn push_orbital(
        &self,
        orbitals: &mut Vec<Orbital>,
        metadata: &mut Metadata,
        coeffs: &mut [Option<f64>],
    ) -> Result<()> {
        let occupancy = metadata
            .occupancy
            .ok_or_else(|| invalid("Molden orbital is missing an occupation number"))?;
        if !(0.0..=2.0).contains(&occupancy) || coeffs.iter().all(Option::is_none) {
            return Err(invalid(
                "Molden orbital needs a physical occupation and real orbital coefficients",
            ));
        }
        if occupancy > 0.0 {
            orbitals.push(
                self.expand_orbital(occupancy, coeffs.iter().map(|weight| weight.unwrap_or(0.0))),
            );
        }
        coeffs.fill(None);
        *metadata = Metadata::default();
        Ok(())
    }
}

#[derive(Default)]
struct Metadata {
    occupancy: Option<f64>,
    spin_seen: bool,
    symmetry_seen: bool,
    energy_seen: bool,
}

fn gaussian_moment(power: u8) -> f64 {
    if power % 2 != 0 {
        return 0.0;
    }
    (1..=power / 2).fold(std::f64::consts::PI.sqrt(), |value, k| {
        value * (f64::from(k) - 0.5)
    })
}

type Polynomial = Vec<([u8; 3], f64)>;

fn angular_polynomials(l: u8, spherical: bool) -> Vec<Polynomial> {
    // Molden's s/p and Cartesian d/f/g basis-index ordering.
    const POWERS: [&[[u8; 3]]; 5] = [
        &[[0, 0, 0]],
        &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        &[
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        &[
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [1, 2, 0],
            [2, 1, 0],
            [2, 0, 1],
            [1, 0, 2],
            [0, 1, 2],
            [0, 2, 1],
            [1, 1, 1],
        ],
        &[
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
            [3, 1, 0],
            [3, 0, 1],
            [1, 3, 0],
            [0, 3, 1],
            [1, 0, 3],
            [0, 1, 3],
            [2, 2, 0],
            [2, 0, 2],
            [0, 2, 2],
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
        ],
    ];
    let powers = POWERS[usize::from(l)];
    if !spherical {
        return powers.iter().map(|&p| vec![(p, 1.0)]).collect();
    }
    let l = i32::from(l);
    std::iter::once(0)
        .chain((1..=l).flat_map(|m| [m, -m]))
        .map(|m| {
            powers
                .iter()
                .filter_map(|&p| {
                    let coefficient = solid_harmonic(l, m, p);
                    (coefficient != 0.0).then_some((p, coefficient))
                })
                .collect()
        })
        .collect()
}

// Real solid harmonics in m = 0,+1,-1,... order with a positive
// leading cosine/sine coefficient. Adapted from the dft-rs AO transform.
fn solid_harmonic(l: i32, m: i32, powers: [u8; 3]) -> f64 {
    let [x, _, z] = powers.map(i32::from);
    let ma = m.abs();
    if z > l - ma || (l - ma - z) % 2 != 0 {
        return 0.0;
    }
    let t = (l - ma - z) / 2;
    let parity = i32::from(m < 0);
    let factorial = |n| (1..=n).map(f64::from).product::<f64>();
    let binomial = |n, k| factorial(n) / (factorial(k) * factorial(n - k));
    (parity..=ma)
        .step_by(2)
        .filter_map(|k| {
            let twice_u = 2 * t + ma - k - x;
            let u = twice_u / 2;
            if twice_u < 0 || twice_u % 2 != 0 || u > t {
                return None;
            }
            let phase = if (t + (k - parity) / 2) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            Some(
                phase
                    * 0.25_f64.powi(t)
                    * binomial(l, t)
                    * binomial(l - t, ma + t)
                    * binomial(t, u)
                    * binomial(ma, k),
            )
        })
        .sum()
}
