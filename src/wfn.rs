// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::wavefunction::{numbers, primitive_powers, Orbital, Primitive};
use crate::{invalid, Atom, Result, Wavefunction};
use std::io::{BufRead, BufReader, Read};

/// Read real Gaussian WFN occupied orbitals. The primitive expansions carry
/// their normalization, and the original documented s through f types apply.
pub fn read_wfn(input: impl Read) -> Result<Wavefunction> {
    let mut lines = BufReader::new(input).lines();
    required_line(&mut lines, "title")?;
    let description = required_line(&mut lines, "counts")?;
    let fields: Vec<_> = description.split_whitespace().collect();
    let ["GAUSSIAN", orbitals, "MOL", "ORBITALS", primitives, "PRIMITIVES", atoms, "NUCLEI"] =
        fields.as_slice()
    else {
        return Err(invalid(
            "unsupported WFN header; expected GAUSSIAN orbital records",
        ));
    };
    let parse_count = |value: &str| {
        value
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| invalid("invalid WFN header count"))
    };
    let orbital_count = parse_count(orbitals)?;
    let primitive_count = parse_count(primitives)?;
    let atom_count = parse_count(atoms)?;
    let mut atoms = Vec::new();
    let mut charges = Vec::new();
    for expected in 1..=atom_count {
        let (atom, charge) = read_atom(&required_line(&mut lines, "nucleus")?, expected)?;
        atoms.push(atom);
        charges.push(charge);
    }
    let centers: Vec<usize> =
        read_fields(&mut lines, primitive_count, "CENTRE ASSIGNMENTS", 20, 3)?;
    let types: Vec<usize> = read_fields(&mut lines, primitive_count, "TYPE ASSIGNMENTS", 20, 3)?;
    let exponents: Vec<f64> = read_fields(&mut lines, primitive_count, "EXPONENTS", 10, 14)?;
    let primitives = centers.into_iter().zip(types).zip(exponents).map(|((center, kind), exponent)| {
        if !(1..=20).contains(&kind) {
            return Err(invalid(format!("unsupported legacy WFN primitive type {kind}; use WFX for higher angular momenta")));
        }
        Ok(Primitive {
            center: center.checked_sub(1).ok_or_else(|| invalid("WFN primitive centers start at one"))?,
            exponent,
            powers: primitive_powers(kind)?,
        })
    }).collect::<Result<Vec<_>>>()?;
    let mut orbitals = Vec::new();
    for _ in 0..orbital_count {
        let line = required_line(&mut lines, "orbital")?;
        if !line.trim_start().starts_with("MO ") {
            return Err(invalid("WFN orbital must start with MO"));
        }
        let (_, occupancy) = line
            .split_once("OCC NO")
            .ok_or_else(|| invalid("missing WFN orbital occupation"))?;
        let (_, occupancy) = occupancy
            .split_once('=')
            .ok_or_else(|| invalid("invalid WFN orbital occupation"))?;
        let occupancy = occupancy
            .split_whitespace()
            .next()
            .ok_or_else(|| invalid("missing WFN occupation value"))?;
        let occupancy: Vec<f64> = numbers(occupancy)?;
        let coefficients = read_fields(&mut lines, primitive_count, "", 0, 16)?;
        orbitals.push(Orbital {
            occupancy: occupancy[0],
            coefficients,
        });
    }
    if required_line(&mut lines, "END DATA")?.trim() != "END DATA" {
        return Err(invalid(
            "WFN contains unexpected data after the orbital records",
        ));
    }
    Wavefunction::new(atoms, primitives, orbitals)?.with_nuclear_data(charges, Vec::new())
}

fn read_atom(line: &str, index: usize) -> Result<(Atom, f64)> {
    let (record, coordinates) = line
        .split_once(')')
        .ok_or_else(|| invalid("invalid WFN nucleus record"))?;
    let (name, center) = record
        .split_once('(')
        .ok_or_else(|| invalid("invalid WFN nucleus label"))?;
    let symbol = name
        .trim_start()
        .split(|character: char| !character.is_ascii_alphabetic())
        .next()
        .unwrap_or("");
    let atomic_number =
        crate::data::atomic_number(symbol).ok_or_else(|| invalid("unknown WFN element symbol"))?;
    let mut center = center.split_whitespace();
    if !matches!(center.next(), Some("CENTRE" | "CENTER"))
        || center.next().and_then(|value| value.parse::<usize>().ok()) != Some(index)
        || center.next().is_some()
    {
        return Err(invalid("WFN nuclei require consecutive center labels"));
    }
    let (coordinates, charge) = coordinates
        .split_once("CHARGE")
        .ok_or_else(|| invalid("missing WFN nuclear charge"))?;
    let charge: Vec<f64> = numbers(
        charge
            .trim_start()
            .strip_prefix('=')
            .ok_or_else(|| invalid("invalid WFN nuclear charge"))?,
    )?;
    let [charge] = charge.as_slice() else {
        return Err(invalid("WFN nucleus requires one charge"));
    };
    let values = numbers::<f64>(coordinates)
        .ok()
        .filter(|values| values.len() == 3)
        .map_or_else(|| fixed_fields(line.get(24..60).unwrap_or(""), 12), Ok)?;
    let position = values
        .try_into()
        .map_err(|_| invalid("WFN nucleus requires three coordinates"))?;
    Ok((
        Atom {
            atomic_number,
            position,
            fragment: 0,
        },
        *charge,
    ))
}

fn required_line(
    lines: &mut impl Iterator<Item = std::io::Result<String>>,
    kind: &str,
) -> Result<String> {
    lines
        .next()
        .ok_or_else(|| invalid(format!("missing WFN {kind} record")))?
        .map_err(Into::into)
}

fn read_fields<T: std::str::FromStr>(
    lines: &mut impl Iterator<Item = std::io::Result<String>>,
    count: usize,
    prefix: &str,
    column: usize,
    width: usize,
) -> Result<Vec<T>> {
    let mut values = Vec::new();
    while values.len() < count {
        let line = required_line(lines, "array")?;
        let body = line
            .trim_start()
            .strip_prefix(prefix)
            .ok_or_else(|| invalid(format!("expected WFN {prefix} array label")))?;
        let next = line
            .get(column..)
            .filter(|body| {
                body.len() >= width && line.get(..column).is_some_and(|left| left.trim() == prefix)
            })
            .map(|body| fixed_fields::<T>(body.trim_end(), width));
        let row = match next {
            Some(Ok(row)) if !row.is_empty() && row.len() <= count - values.len() => row,
            _ => numbers::<T>(body)?,
        };
        if row.is_empty() || row.len() > count - values.len() {
            return Err(invalid("WFN array does not match its declared length"));
        }
        values.extend(row);
    }
    Ok(values)
}

fn fixed_fields<T: std::str::FromStr>(text: &str, width: usize) -> Result<Vec<T>> {
    if !text.is_ascii() {
        return Err(invalid("WFN numeric records must be ASCII"));
    }
    text.as_bytes()
        .chunks(width)
        .map(|field| {
            let text = std::str::from_utf8(field).expect("ASCII");
            let mut values: Vec<T> = numbers(text)?;
            if values.len() != 1 {
                return Err(invalid("invalid WFN fixed-width numeric field"));
            }
            Ok(values.pop().expect("one field"))
        })
        .collect()
}
