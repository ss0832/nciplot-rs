// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::data::{add, norm, scale, sub};
use crate::{invalid, Dataset, DensityKind, DensitySample, Result, Vec3};
use std::fmt;
use std::str::FromStr;

/// The atom-gradient convention used for the scalar fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Nci,
    /// The original promolecular independent gradient model.
    Igm,
    /// The inverted stockholder-weight derivative used for the published maps.
    Igmh,
    /// The mathematical derivative of the Hirshfeld-partitioned molecular density.
    IgmhHirshfeld,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nci => "nci",
            Self::Igm => "igm",
            Self::Igmh => "igmh",
            Self::IgmhHirshfeld => "igmh-hirshfeld",
        })
    }
}

impl FromStr for Method {
    type Err = crate::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "nci" => Ok(Self::Nci),
            "igm" => Ok(Self::Igm),
            "igmh" => Ok(Self::Igmh),
            "igmh-hirshfeld" => Ok(Self::IgmhHirshfeld),
            _ => Err(invalid(format!("unknown analysis method {value}"))),
        }
    }
}

impl Method {
    pub fn default_descriptor(self, dataset: &Dataset) -> Descriptor {
        if self == Self::Nci {
            Descriptor::Rdg
        } else if dataset
            .atoms()
            .windows(2)
            .any(|atoms| atoms[0].fragment != atoms[1].fragment)
        {
            Descriptor::Inter
        } else {
            Descriptor::Total
        }
    }
}

/// A field selectable for volumetric output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Descriptor {
    Rdg,
    Total,
    Inter,
    Intra,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Rdg => "rdg",
            Self::Total => "total",
            Self::Inter => "inter",
            Self::Intra => "intra",
        })
    }
}

impl FromStr for Descriptor {
    type Err = crate::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "rdg" => Ok(Self::Rdg),
            "total" => Ok(Self::Total),
            "inter" => Ok(Self::Inter),
            "intra" => Ok(Self::Intra),
            _ => Err(invalid(format!("unknown descriptor {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GradientDifference {
    /// All gradient differences in electrons per Bohr to the fourth power.
    pub total: f64,
    pub inter: f64,
    pub intra: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct InteractionPoint {
    pub density: f64,
    /// Middle Cartesian Hessian eigenvalue, in electrons per Bohr to the fifth power.
    pub lambda2: f64,
    /// Sign of the middle Hessian eigenvalue times the electron density.
    pub signed_density: f64,
    /// The dimensionless reduced density gradient. Undefined at zero density.
    pub rdg: Option<f64>,
    pub delta_g: Option<GradientDifference>,
}

impl InteractionPoint {
    pub fn value(self, descriptor: Descriptor) -> Option<f64> {
        match descriptor {
            Descriptor::Rdg => self.rdg,
            Descriptor::Total => self.delta_g.map(|field| field.total),
            Descriptor::Inter => self.delta_g.map(|field| field.inter),
            Descriptor::Intra => self.delta_g.map(|field| field.intra),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub method: Method,
    pub points: Vec<InteractionPoint>,
    /// Maximum density closure defect for IGMH in electrons per cubic Bohr.
    pub density_residual: Option<f64>,
    /// Maximum norm of the IGMH gradient closure defect in electrons per Bohr^4.
    pub gradient_residual: Option<f64>,
    /// Count of points where the density is exactly zero.
    pub zero_density_points: usize,
}

/// Evaluate the descriptors with the analytic input derivatives. Each output
/// index corresponds to the same index in the dataset.
pub fn analyze(dataset: &Dataset, method: Method) -> Result<Analysis> {
    let hirshfeld = matches!(method, Method::Igmh | Method::IgmhHirshfeld);
    if hirshfeld && dataset.density_kind() != DensityKind::Molecular {
        return Err(invalid("IGMH requires a molecular density field"));
    }
    let proatoms = if method == Method::Nci {
        None
    } else {
        Some(
            dataset
                .proatom_samples()
                .ok_or_else(|| invalid("IGM and IGMH require neutral proatom samples"))?,
        )
    };
    let mut labels: Vec<_> = dataset.atoms().iter().map(|atom| atom.fragment).collect();
    labels.sort_unstable();
    labels.dedup();
    let groups: Vec<_> = dataset
        .atoms()
        .iter()
        .map(|atom| {
            labels
                .binary_search(&atom.fragment)
                .expect("existing label")
        })
        .collect();
    let mut fragment_gradients = vec![[0.0; 3]; labels.len()];
    let mut analysis = Analysis {
        method,
        points: Vec::with_capacity(dataset.grid().point_count()),
        density_residual: hirshfeld.then_some(0.0),
        gradient_residual: hirshfeld.then_some(0.0),
        zero_density_points: 0,
    };
    let atom_count = groups.len();
    for (index, &sample) in dataset.density_samples().iter().enumerate() {
        let mut point = nci_point(sample);
        if !point.lambda2.is_finite() || point.rdg.is_some_and(|rdg| !rdg.is_finite()) {
            return Err(invalid(format!("NCI arithmetic overflow at node {index}")));
        }
        analysis.zero_density_points += usize::from(point.rdg.is_none());
        if let Some(proatoms) = proatoms {
            let proatoms = &proatoms[index * atom_count..(index + 1) * atom_count];
            let mut density_sum: f64 = 0.0;
            let mut gradient_sum = [0.0; 3];
            let mut atom_norms = 0.0;
            fragment_gradients.fill([0.0; 3]);

            // Rescale the promolecule density before forming weights. This
            // preserves usable weights in the low-density tails.
            let max_density = if hirshfeld {
                proatoms.iter().map(|atom| atom.density).fold(0.0, f64::max)
            } else {
                0.0
            };
            if hirshfeld
                && max_density == 0.0
                && (sample.density != 0.0 || sample.gradient != [0.0; 3])
            {
                return Err(invalid(format!(
                    "nonzero molecular field with zero promolecule density at node {index}"
                )));
            }
            let denominator: f64 = if max_density > 0.0 {
                proatoms.iter().map(|atom| atom.density / max_density).sum()
            } else {
                0.0
            };
            let density_over_max = if max_density > 0.0 && sample.density != 0.0 {
                sample.density / max_density / denominator
            } else {
                0.0
            };
            let weighted_prograd: Vec3 = if density_over_max == 0.0 {
                [0.0; 3]
            } else {
                proatoms.iter().fold([0.0; 3], |sum, atom| {
                    add(sum, scale(atom.gradient, density_over_max))
                })
            };
            let sign = if method == Method::Igmh { -1.0 } else { 1.0 };
            for (atom, &group) in proatoms.iter().zip(&groups) {
                let vector = if method == Method::Igm {
                    atom.gradient
                } else if max_density == 0.0 {
                    [0.0; 3]
                } else {
                    let weight = (atom.density / max_density) / denominator;
                    density_sum += sample.density * weight;
                    // Product rule with either sign for the derivative of w_A.
                    let density_times_dw = sub(
                        scale(atom.gradient, density_over_max),
                        scale(weighted_prograd, weight),
                    );
                    add(
                        scale(sample.gradient, weight),
                        scale(density_times_dw, sign),
                    )
                };
                fragment_gradients[group] = add(fragment_gradients[group], vector);
                gradient_sum = add(gradient_sum, vector);
                atom_norms += norm(vector);
            }
            let fragment_norms: f64 = fragment_gradients.iter().copied().map(norm).sum();
            let total_norm = norm(gradient_sum);
            if !(atom_norms.is_finite() && fragment_norms.is_finite() && total_norm.is_finite()) {
                return Err(invalid(format!(
                    "gradient arithmetic overflow at node {index}"
                )));
            }
            // Clamp only roundoff excursions that violate the triangle inequality.
            let inter = (fragment_norms - total_norm).max(0.0);
            let intra = (atom_norms - fragment_norms).max(0.0);
            point.delta_g = Some(GradientDifference {
                total: inter + intra,
                inter,
                intra,
            });
            if let (Some(density), Some(gradient)) = (
                &mut analysis.density_residual,
                &mut analysis.gradient_residual,
            ) {
                let density_defect = (density_sum - sample.density).abs();
                let gradient_defect = norm(sub(gradient_sum, sample.gradient));
                if !(density_defect.is_finite() && gradient_defect.is_finite()) {
                    return Err(invalid(format!(
                        "partition closure overflow at node {index}"
                    )));
                }
                *density = density.max(density_defect);
                *gradient = gradient.max(gradient_defect);
            }
        }
        analysis.points.push(point);
    }
    Ok(analysis)
}

fn nci_point(sample: DensitySample) -> InteractionPoint {
    let lambda2 = middle_eigenvalue(sample.hessian);
    let signed_density = if lambda2 == 0.0 {
        0.0
    } else {
        sample.density.copysign(lambda2)
    };
    let rdg = (sample.density > 0.0).then(|| {
        let denominator = 2.0 * (3.0 * std::f64::consts::PI.powi(2)).cbrt();
        (norm(sample.gradient) / sample.density) / denominator / sample.density.cbrt()
    });
    InteractionPoint {
        density: sample.density,
        lambda2,
        signed_density,
        rdg,
        delta_g: None,
    }
}

fn middle_eigenvalue(hessian: [f64; 6]) -> f64 {
    // Work on a scaled symmetric matrix so the Jacobi rotations do not
    // overflow in their shifts, including in high-density core regions.
    let magnitude = hessian.into_iter().map(f64::abs).fold(0.0, f64::max);
    if magnitude == 0.0 {
        return 0.0;
    }
    let [xx, xy, xz, yy, yz, zz] = hessian.map(|value| value / magnitude);
    let mut a = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
    for _ in 0..12 {
        for (p, q, r) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)] {
            let coupling = a[p][q];
            if coupling == 0.0 {
                continue;
            }
            let tau = (a[q][q] - a[p][p]) / (2.0 * coupling);
            let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
            let c = t.hypot(1.0).recip();
            let s = t * c;
            a[p][p] -= t * coupling;
            a[q][q] += t * coupling;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            let pr = a[p][r];
            let qr = a[q][r];
            a[p][r] = c * pr - s * qr;
            a[q][r] = s * pr + c * qr;
            a[r][p] = a[p][r];
            a[r][q] = a[q][r];
        }
    }
    let mut eigenvalues = [a[0][0], a[1][1], a[2][2]];
    eigenvalues.sort_by(f64::total_cmp);
    eigenvalues[1] * magnitude
}
