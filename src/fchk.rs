// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::molden::Basis;
use crate::wavefunction::MatrixDensity;
use crate::{invalid, Atom, Result, Wavefunction};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read},
    str::FromStr,
};

const FIELDS: &[&str] = &[
    "Number of atoms",
    "Number of electrons",
    "Number of alpha electrons",
    "Number of beta electrons",
    "Number of basis functions",
    "Atomic numbers",
    "Nuclear charges",
    "Current cartesian coordinates",
    "Shell types",
    "Number of primitives per shell",
    "Shell to atom map",
    "Primitive exponents",
    "Contraction coefficients",
    "P(S=P) Contraction coefficients",
    "Coordinates of each shell",
    "Alpha MO coefficients",
    "Beta MO coefficients",
    "Total SCF Density",
    "Alpha Orbital Occupation Numbers",
    "Beta Orbital Occupation Numbers",
];

/// Read the SCF orbital density of a real molecular formatted checkpoint.
pub fn read_fchk(input: impl Read) -> Result<Wavefunction> {
    let mut lines = BufReader::new(input).lines();
    line(&mut lines)?;
    let job = line(&mut lines)?;
    if job
        .split_whitespace()
        .any(|word| word.starts_with("GHF") || word.starts_with("GKS"))
        || job.is_empty()
    {
        return Err(invalid(
            "FCHK input requires real spatial or collinear SCF orbitals",
        ));
    }
    let mut fields = BTreeMap::new();
    while let Some(record) = lines.next() {
        let record = record?;
        let Some((key, tail)) = record.get(..40).zip(record.get(40..)) else {
            continue;
        };
        let key = key.trim();
        let Some(&key) = FIELDS.iter().find(|&&name| name.eq_ignore_ascii_case(key)) else {
            if matches!(
                key,
                "Translation vectors"
                    | "Number of translation vectors"
                    | "Alpha MO coefficients imaginary"
                    | "Beta MO coefficients imaginary"
            ) {
                return Err(invalid(
                    "FCHK lattice and complex coefficient extensions are unsupported",
                ));
            }
            continue;
        };
        let tail = tail.trim();
        let (kind, values) = tail
            .split_at_checked(1)
            .ok_or_else(|| invalid("missing FCHK data descriptor"))?;
        if !matches!(kind, "I" | "R") {
            return Err(invalid(format!("unsupported FCHK data type for {key}")));
        }
        let values = values.trim_start();
        let data = if let Some(count) = values.strip_prefix("N=") {
            let count = count
                .trim()
                .parse::<usize>()
                .map_err(|_| invalid(format!("invalid FCHK array count for {key}")))?;
            let mut values = Vec::new();
            while values.len() < count {
                let next = line(&mut lines)?;
                for token in next.split_whitespace() {
                    values.push(token.to_owned());
                }
                if values.len() > count {
                    return Err(invalid(format!(
                        "FCHK array exceeds declared count for {key}"
                    )));
                }
            }
            values
        } else {
            vec![values.trim().to_owned()]
        };
        if fields.insert(key, data).is_some() {
            return Err(invalid(format!("duplicate FCHK field {key}")));
        }
    }
    let fields = Fields(fields);
    let atoms = fields.atoms()?;
    let charges = fields.values::<f64>("Nuclear charges")?;
    let basis = fields.basis(&atoms)?;
    let count = basis.function_count();
    if count == 0 || count != fields.scalar::<usize>("Number of basis functions")? {
        return Err(invalid(
            "FCHK basis function count does not match the shell conventions",
        ));
    }
    let alpha = fields.scalar::<usize>("Number of alpha electrons")?;
    let beta = fields.scalar::<usize>("Number of beta electrons")?;
    let electrons = fields.scalar::<usize>("Number of electrons")?;
    if alpha.checked_add(beta) != Some(electrons) {
        return Err(invalid(
            "FCHK alpha/beta counts disagree with the total electron count",
        ));
    }
    if fields.0.contains_key("Total SCF Density") {
        let matrix = MatrixDensity {
            basis: basis.functions,
            lower_triangle: fields.values::<f64>("Total SCF Density")?,
            electrons,
        };
        return Wavefunction::new_matrix(atoms, basis.primitives, matrix)?
            .with_nuclear_data(charges, Vec::new());
    }
    if fields.0.contains_key("Alpha Orbital Occupation Numbers")
        || fields.0.contains_key("Beta Orbital Occupation Numbers")
    {
        return Err(invalid(
            "FCHK files with explicit occupations require Total SCF Density",
        ));
    }
    let spin_beta = fields.0.contains_key("Beta MO coefficients");
    let mut orbitals = Vec::new();
    for (label, occupied) in [
        ("Alpha MO coefficients", alpha),
        ("Beta MO coefficients", beta),
    ] {
        if !spin_beta && label.starts_with("Beta") {
            continue;
        }
        let coefficients = fields.values::<f64>(label)?;
        if coefficients.len() % count != 0
            || coefficients.len() / count < occupied
            || coefficients.iter().any(|value| !value.is_finite())
            || (!spin_beta && beta > alpha)
        {
            return Err(invalid(format!(
                "FCHK coefficient dimensions disagree with {label} occupations"
            )));
        }
        for (index, weights) in coefficients.chunks_exact(count).take(occupied).enumerate() {
            let occupation = if !spin_beta && index < beta { 2.0 } else { 1.0 };
            orbitals.push(basis.expand_orbital(occupation, weights.iter().copied()));
        }
    }
    Wavefunction::new(atoms, basis.primitives, orbitals)?.with_nuclear_data(charges, Vec::new())
}

struct Fields(BTreeMap<&'static str, Vec<String>>);

impl Fields {
    fn values<T: FromStr>(&self, label: &str) -> Result<Vec<T>> {
        self.0
            .get(label)
            .ok_or_else(|| invalid(format!("missing FCHK field {label}")))?
            .iter()
            .map(|token| {
                token
                    .replace(['D', 'd'], "e")
                    .parse()
                    .map_err(|_| invalid(format!("invalid numeric value in FCHK field {label}")))
            })
            .collect()
    }

    fn scalar<T: FromStr>(&self, label: &str) -> Result<T> {
        let mut values = self.values::<T>(label)?;
        if values.len() != 1 {
            return Err(invalid(format!("FCHK value {label} must be scalar")));
        }
        Ok(values.pop().expect("one value"))
    }

    fn atoms(&self) -> Result<Vec<Atom>> {
        let atoms = self.scalar::<usize>("Number of atoms")?;
        let elements = self.values::<u8>("Atomic numbers")?;
        let positions = self.values::<f64>("Current cartesian coordinates")?;
        if elements.len() != atoms || atoms.checked_mul(3) != Some(positions.len()) {
            return Err(invalid("FCHK atom arrays disagree with the atom count"));
        }
        Ok(elements
            .into_iter()
            .zip(positions.chunks_exact(3))
            .map(|(atomic_number, position)| Atom {
                atomic_number,
                position: position.try_into().expect("three coordinates"),
                fragment: 0,
            })
            .collect())
    }

    fn basis(&self, atoms: &[Atom]) -> Result<Basis> {
        let shells = self.values::<i8>("Shell types")?;
        let sizes = self.values::<usize>("Number of primitives per shell")?;
        let centers = self.values::<usize>("Shell to atom map")?;
        let exponents = self.values::<f64>("Primitive exponents")?;
        let coefficients = self.values::<f64>("Contraction coefficients")?;
        let sp = if shells.contains(&-1) {
            self.values::<f64>("P(S=P) Contraction coefficients")?
        } else {
            vec![0.0; exponents.len()]
        };
        let positions = if self.0.contains_key("Coordinates of each shell") {
            let positions = self.values::<f64>("Coordinates of each shell")?;
            if shells.len().checked_mul(3) != Some(positions.len()) {
                return Err(invalid(
                    "FCHK shell coordinate count disagrees with the shell count",
                ));
            }
            Some(positions)
        } else {
            None
        };
        let primitive_count = sizes
            .iter()
            .try_fold(0usize, |sum, &count| sum.checked_add(count));
        if sizes.len() != shells.len()
            || centers.len() != shells.len()
            || sizes.contains(&0)
            || primitive_count != Some(exponents.len())
            || coefficients.len() != exponents.len()
            || sp.len() != exponents.len()
        {
            return Err(invalid(
                "FCHK Gaussian arrays disagree with the shell counts",
            ));
        }
        let mut basis = Basis::default();
        let mut offset = 0;
        for (index, ((kind, count), center)) in
            shells.into_iter().zip(sizes).zip(centers).enumerate()
        {
            let center = center
                .checked_sub(1)
                .filter(|&center| center < atoms.len())
                .ok_or_else(|| invalid("invalid FCHK shell center"))?;
            if positions.as_ref().is_some_and(|xyz| {
                (0..3).any(|axis| {
                    let point = xyz[index * 3 + axis];
                    !point.is_finite() || (point - atoms[center].position[axis]).abs() > 1.0e-7
                })
            }) {
                return Err(invalid("displaced FCHK shell centers are unsupported"));
            }
            let parameters = (offset..offset + count)
                .map(|i| vec![exponents[i], coefficients[i], sp[i]])
                .collect::<Vec<_>>();
            if parameters.iter().any(|values| {
                !values[0].is_finite()
                    || values[0] <= 0.0
                    || values.iter().any(|value| !value.is_finite())
            }) {
                return Err(invalid(
                    "invalid FCHK Gaussian exponents or contraction coefficients",
                ));
            }
            if kind == -1 {
                basis.add_shell(center, 0, false, 1, &parameters)?;
                basis.add_shell(center, 1, false, 2, &parameters)?;
            } else if (-3..=3).contains(&kind) {
                basis.add_shell(center, kind.unsigned_abs(), kind < 0, 1, &parameters)?;
            } else {
                return Err(invalid(format!(
                    "unsupported FCHK shell type {kind}; use WFX for higher angular momenta"
                )));
            }
            offset += count;
        }
        Ok(basis)
    }
}

fn line(lines: &mut impl Iterator<Item = std::io::Result<String>>) -> Result<String> {
    lines
        .next()
        .ok_or_else(|| invalid("truncated FCHK record"))?
        .map_err(Into::into)
}
