// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::data::{norm, sub, ELEMENTS};
use crate::io::{validate_analysis, validate_descriptor};
use crate::{invalid, Analysis, Dataset, Descriptor, Result, Vec3};
use std::collections::HashMap;
use std::io::Write;

const PAGE: &str = include_str!("viewer.html");
const DATA_MARKER: &str = "__NCIPLOT_DATA__";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceDensity {
    Unsigned,
    Signed,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewerOptions {
    pub descriptor: Descriptor,
    /// Dimensionless for RDG, electrons per Bohr^4 for a gradient difference.
    pub isovalue: f64,
    /// Density quantity used to clip the surface on initial load.
    pub surface_density: SurfaceDensity,
    /// Inclusive clipping bounds, in electrons per cubic Bohr.
    pub surface_density_min: f64,
    pub surface_density_max: f64,
    /// Endpoints of the surface colour ramp, in electrons per cubic Bohr.
    pub color_min: f64,
    pub color_max: f64,
    /// Symmetric signed-density scatter x axis, in electrons per cubic Bohr.
    pub scatter_density_limit: f64,
}

impl ViewerOptions {
    pub fn new(descriptor: Descriptor) -> Self {
        Self::for_density(descriptor, SurfaceDensity::Unsigned)
    }

    pub fn for_density(descriptor: Descriptor, surface_density: SurfaceDensity) -> Self {
        let nci = descriptor == Descriptor::Rdg;
        let (min, max) = match surface_density {
            SurfaceDensity::Signed => (f64::NEG_INFINITY, f64::INFINITY),
            SurfaceDensity::Unsigned if nci => (0.0, 0.08),
            SurfaceDensity::Unsigned => (0.0, f64::INFINITY),
        };
        Self {
            descriptor,
            isovalue: if nci { 0.5 } else { 0.005 },
            surface_density,
            surface_density_min: min,
            surface_density_max: max,
            color_min: -0.04,
            color_max: 0.04,
            scatter_density_limit: 0.08,
        }
    }

    fn validate(self) -> Result<()> {
        if [self.isovalue, self.scatter_density_limit]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
            || !self.color_min.is_finite()
            || !self.color_max.is_finite()
            || self.color_min >= self.color_max
            || (self.surface_density == SurfaceDensity::Unsigned && self.surface_density_min < 0.0)
            || self
                .surface_density_max
                .partial_cmp(&self.surface_density_min)
                != Some(std::cmp::Ordering::Greater)
        {
            return Err(invalid("viewer needs a positive isovalue and scatter scale, increasing colour and density bounds, and nonnegative bounds for unsigned density"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Node {
    index: usize,
    pos: Vec3,
    field: f64,
    density: f64,
    signed_density: f64,
}

const TETS: [[usize; 4]; 6] = [
    [0, 4, 6, 7],
    [0, 6, 2, 7],
    [0, 2, 3, 7],
    [0, 3, 1, 7],
    [0, 1, 5, 7],
    [0, 5, 4, 7],
];

/// Write a single offline HTML document with a rotatable indexed isosurface,
/// adjustable density bounds, and full/detail scatter views.
pub fn write_html(
    mut out: impl Write,
    dataset: &Dataset,
    analysis: &Analysis,
    options: ViewerOptions,
) -> Result<()> {
    validate_analysis(dataset, analysis)?;
    validate_descriptor(analysis, options.descriptor)?;
    options.validate()?;
    let Mesh {
        vertices, faces, ..
    } = surface(dataset, analysis, options)?;
    let (head, tail) = PAGE.split_once(DATA_MARKER).expect("viewer data marker");
    out.write_all(head.as_bytes())?;

    let rdg_surface = options.descriptor == Descriptor::Rdg;
    let delta = if rdg_surface {
        Descriptor::Total
    } else {
        options.descriptor
    };
    let scatter = || {
        analysis.points.iter().filter(|point| {
            point.signed_density.abs() <= options.scatter_density_limit
                && (point.rdg.is_some() || point.delta_g.is_some())
        })
    };
    let delta_max = scatter()
        .filter_map(|point| point.value(delta))
        .fold(0.0, f64::max);
    let delta_limit = if delta_max > 0.0 {
        (1.06 * delta_max).min(f64::MAX)
    } else {
        1.0
    };
    let limit = |value: f64| {
        if value.is_finite() {
            value.to_string()
        } else {
            "null".to_owned()
        }
    };
    let surface_min = limit(options.surface_density_min);
    let surface_max = limit(options.surface_density_max);
    write!(
        out,
        "{{\"method\":\"{}\",\"field\":\"{}\",\"isovalue\":{},\"colorMin\":{},\"colorMax\":{},\"scatterDensityLimit\":{},\"igmhLimit\":{},\"nciDetailLimit\":2,\"surfaceDensityMin\":{},\"surfaceDensityMax\":{},\"signedSurfaceDensity\":{},\"hasDelta\":{},\"atoms\":[",
        analysis.method, options.descriptor, options.isovalue, options.color_min, options.color_max,
        options.scatter_density_limit, delta_limit, surface_min, surface_max,
        options.surface_density == SurfaceDensity::Signed,
        analysis.points.iter().any(|point| point.delta_g.is_some()),
    )?;
    for (index, atom) in dataset.atoms().iter().enumerate() {
        comma(&mut out, index)?;
        let [x, y, z] = atom.position;
        write!(
            out,
            "[{},{x},{y},{z},\"{}\"]",
            atom.atomic_number,
            ELEMENTS[usize::from(atom.atomic_number)]
        )?;
    }
    write!(out, "],\"bonds\":[")?;
    for (index, [a, b]) in bonds(dataset).into_iter().enumerate() {
        comma(&mut out, index)?;
        write!(out, "[{a},{b}]")?;
    }

    write!(out, "],\"vertices\":")?;
    packed(
        &mut out,
        vertices.into_iter().flatten().flat_map(f32::to_le_bytes),
    )?;
    write!(out, ",\"faces\":")?;
    packed(
        &mut out,
        faces.into_iter().flatten().flat_map(u32::to_le_bytes),
    )?;
    write!(out, ",\"scatter\":")?;
    packed(
        &mut out,
        scatter()
            .flat_map(|point| {
                [
                    point.signed_density,
                    point.value(delta).unwrap_or(f64::NAN),
                    point.rdg.unwrap_or(f64::NAN),
                ]
            })
            .flat_map(f64::to_le_bytes),
    )?;
    write!(out, "}}")?;
    out.write_all(tail.as_bytes())?;
    Ok(())
}

fn comma(out: &mut impl Write, index: usize) -> std::io::Result<()> {
    if index != 0 {
        out.write_all(b",")?;
    }
    Ok(())
}

fn packed(out: &mut impl Write, mut bytes: impl Iterator<Item = u8>) -> std::io::Result<()> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut buffer = [0u8; 4096];
    let mut offset = 0;
    out.write_all(b"\"")?;
    while let Some(a) = bytes.next() {
        let b = bytes.next();
        let c = bytes.next();
        let bb = b.unwrap_or(0);
        let cc = c.unwrap_or(0);
        buffer[offset] = ALPHABET[usize::from(a >> 2)];
        buffer[offset + 1] = ALPHABET[usize::from((a & 3) << 4 | bb >> 4)];
        buffer[offset + 2] = if b.is_some() {
            ALPHABET[usize::from((bb & 15) << 2 | cc >> 6)]
        } else {
            b'='
        };
        buffer[offset + 3] = if c.is_some() {
            ALPHABET[usize::from(cc & 63)]
        } else {
            b'='
        };
        offset += 4;
        if offset == buffer.len() {
            out.write_all(&buffer)?;
            offset = 0;
        }
    }
    out.write_all(&buffer[..offset])?;
    out.write_all(b"\"")
}

#[derive(Default)]
struct Mesh {
    vertices: Vec<[f32; 5]>,
    faces: Vec<[u32; 3]>,
    indices: HashMap<[usize; 2], u32>,
}

impl Mesh {
    fn crossing(&mut self, a: Node, b: Node, level: f64) -> Result<u32> {
        let (a, b) = if a.index < b.index { (a, b) } else { (b, a) };
        let mut key = [a.index, b.index];
        let t = if level == a.field {
            key = [a.index; 2];
            0.0
        } else if level == b.field {
            key = [b.index; 2];
            1.0
        } else {
            (level - a.field) / (b.field - a.field)
        };
        if let Some(&index) = self.indices.get(&key) {
            return Ok(index);
        }
        let blend = |a: f64, b: f64| ((1.0 - t) * a + t * b) as f32;
        let [x, y, z] = std::array::from_fn(|axis| blend(a.pos[axis], b.pos[axis]));
        let vertex = [
            x,
            y,
            z,
            blend(a.signed_density, b.signed_density),
            blend(a.density, b.density),
        ];
        if !vertex.into_iter().all(f32::is_finite) {
            return Err(invalid(
                "surface coordinates and densities exceed the browser precision range",
            ));
        }
        let index = u32::try_from(self.vertices.len())
            .map_err(|_| invalid("isosurface exceeds the browser index range"))?;
        self.vertices.push(vertex);
        self.indices.insert(key, index);
        Ok(index)
    }

    fn add_face(&mut self, indices: [u32; 3]) {
        let [a, b, c] = indices;
        if a != b && a != c && b != c {
            self.faces.push(indices);
        }
    }

    fn section(&mut self, points: [Node; 4], level: f64) -> Result<()> {
        let above = points.iter().copied().filter(|point| point.field >= level);
        let below = points.iter().copied().filter(|point| point.field < level);
        let mut top = above.clone();
        let mut bottom = below.clone();
        match above.count() {
            1 => {
                let a = top.next().expect("one above");
                let mut face = [0; 3];
                for (vertex, b) in face.iter_mut().zip(bottom) {
                    *vertex = self.crossing(a, b, level)?;
                }
                self.add_face(face);
            }
            2 => {
                let a = top.next().expect("two above");
                let b = top.next().expect("two above");
                let c = bottom.next().expect("two below");
                let d = bottom.next().expect("two below");
                let ac = self.crossing(a, c, level)?;
                let ad = self.crossing(a, d, level)?;
                let bc = self.crossing(b, c, level)?;
                let bd = self.crossing(b, d, level)?;
                self.add_face([ac, ad, bd]);
                self.add_face([ac, bd, bc]);
            }
            3 => {
                let b = bottom.next().expect("one below");
                let mut face = [0; 3];
                for (vertex, a) in face.iter_mut().zip(top) {
                    *vertex = self.crossing(a, b, level)?;
                }
                self.add_face(face);
            }
            _ => {}
        }
        Ok(())
    }
}

fn surface(dataset: &Dataset, analysis: &Analysis, options: ViewerOptions) -> Result<Mesh> {
    let grid = dataset.grid();
    let [nx, ny, nz] = grid.shape();
    let mut mesh = Mesh::default();
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            for k in 0..nz - 1 {
                let corners = std::array::from_fn::<_, 8, _>(|corner| {
                    let indices = [i + corner / 4, j + corner / 2 % 2, k + corner % 2];
                    let index = grid.linear(indices);
                    let point = analysis.points[index];
                    point.value(options.descriptor).map(|field| Node {
                        index,
                        pos: grid.point(indices),
                        field,
                        density: point.density,
                        signed_density: point.signed_density,
                    })
                });
                if corners
                    .iter()
                    .flatten()
                    .all(|point| point.field >= options.isovalue)
                    || corners
                        .iter()
                        .flatten()
                        .all(|point| point.field < options.isovalue)
                {
                    continue;
                }
                for tet in TETS {
                    if let [Some(a), Some(b), Some(c), Some(d)] = tet.map(|index| corners[index]) {
                        mesh.section([a, b, c, d], options.isovalue)?;
                    }
                }
            }
        }
    }
    Ok(mesh)
}

fn covalent_radius(atomic_number: u8) -> Option<f64> {
    let angstrom = match atomic_number {
        1 => 0.31,
        5 => 0.84,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        14 => 1.11,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        35 => 1.20,
        53 => 1.39,
        _ => return None,
    };
    Some(angstrom / 0.529_177_210_903)
}

fn bonds(dataset: &Dataset) -> Vec<[usize; 2]> {
    // Spatial bins avoid an all-pairs search in macromolecules.
    let width = 7.0;
    let mut bins: HashMap<_, Vec<usize>> = HashMap::new();
    let mut bonds = Vec::new();
    for (index, atom) in dataset.atoms().iter().enumerate() {
        let Some(radius) = covalent_radius(atom.atomic_number) else {
            continue;
        };
        let bin = atom
            .position
            .map(|coordinate| (coordinate / width).floor() as i64);
        for x in -1i64..=1 {
            for y in -1i64..=1 {
                for z in -1i64..=1 {
                    let neighbor = [x, y, z];
                    let keys = std::array::from_fn(|axis| bin[axis].saturating_add(neighbor[axis]));
                    if let Some(indices) = bins.get(&(keys, atom.fragment)) {
                        for &other in indices {
                            let b = dataset.atoms()[other];
                            let threshold = 1.25
                                * (radius
                                    + covalent_radius(b.atomic_number)
                                        .expect("binned atom radius"));
                            let distance = norm(sub(atom.position, b.position));
                            if distance > 0.2 && distance <= threshold {
                                bonds.push([other, index]);
                            }
                        }
                    }
                }
            }
        }
        bins.entry((bin, atom.fragment)).or_default().push(index);
    }
    bonds
}
