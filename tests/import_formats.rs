// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use nciplot_rs::{
    analyze, read_expanded, read_fchk, read_wfn, read_wfx, DensityModel, DensitySample, Method,
    Wavefunction,
};
use std::fmt::Write;

const POINT: [f64; 3] = [0.61, 0.47, 0.91];
const CENTERS: [[f64; 3]; 2] = [[0.29, -0.17, 0.31], [1.18, 0.36, -0.21]];
const EXPONENTS: [f64; 2] = [0.64, 0.91];
const WIDTH: [f64; 3] = [2.4, 0.31, -0.18];
const COMPONENTS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

fn gaussian(at: [f64; 3], center: [f64; 3], exponent: f64) -> f64 {
    (-exponent
        * at.into_iter()
            .zip(center)
            .map(|(x, c)| (x - c).powi(2))
            .sum::<f64>())
    .exp()
}

fn components(sample: DensitySample) -> [f64; 10] {
    let [x, y, z] = sample.gradient;
    let [xx, xy, xz, yy, yz, zz] = sample.hessian;
    [sample.density, x, y, z, xx, xy, xz, yy, yz, zz]
}

fn fitting(wf: &Wavefunction, label: &str, density: impl Fn([f64; 3]) -> f64) {
    let values = wf.density_at(POINT).unwrap();
    let rhs = density(POINT);
    assert!((values.density - rhs).abs() < 3.0e-16);
    let production: Vec<_> = values.gradient.into_iter().chain(values.hessian).collect();
    let steps = [0.028_f64, 0.018, 0.011, 0.007];
    let x = steps.map(f64::ln);
    let mean = |v: [f64; 4]| v.into_iter().sum::<f64>() / 4.0;
    let dx = x.map(|v| v - mean(x));
    let ssx: f64 = dx.map(|v| v * v).into_iter().sum();
    let independent = |h: f64| {
        let f = |i: usize, di: f64, j: usize, dj: f64| {
            let mut p = POINT;
            p[i] += di;
            p[j] += dj;
            density(p)
        };
        std::array::from_fn::<_, 9, _>(|id| {
            if id < 3 {
                let i = id;
                (f(i, h, i, 0.0) - f(i, -h, i, 0.0)) / (2.0 * h)
            } else {
                let (i, j) = COMPONENTS[id - 3];
                if i == j {
                    (f(i, h, i, 0.0) - 2.0 * rhs + f(i, -h, i, 0.0)) / (h * h)
                } else {
                    (f(i, h, j, h) - f(i, -h, j, h) - f(i, h, j, -h) + f(i, -h, j, -h))
                        / (4.0 * h * h)
                }
            }
        })
    };
    let ratios = steps.map(independent);
    for (i, production) in production.into_iter().enumerate() {
        let err = ratios.map(|rhs| (rhs[i] - production).abs());
        assert!(err.into_iter().all(|e| e > 0.0));
        let y = err.map(f64::ln);
        let dy = y.map(|v| v - mean(y));
        let p = dx.into_iter().zip(dy).map(|(a, b)| a * b).sum::<f64>() / ssx;
        let c = (mean(y) - p * mean(x)).exp();
        let residual: f64 = dx
            .into_iter()
            .zip(dy)
            .map(|(a, b)| (b - p * a).powi(2))
            .sum();
        let rsq = 1.0 - residual / dy.map(|d| d * d).into_iter().sum::<f64>();
        println!("{label} {i} p={p:.8} C={c:.8e} R2={rsq:.10} h=[0.007, 0.028] errors={err:?}");
        assert!((p - 2.0).abs() < 0.05 && rsq > 0.9999);
    }
}

fn fchk_atom_data() -> String {
    let mut out = String::from(
        "H2
SP        RHF                           Custom
",
    );
    for (key, value) in [
        ("Number of atoms", 2),
        ("Number of electrons", 2),
        ("Number of alpha electrons", 1),
        ("Number of beta electrons", 1),
        ("Number of basis functions", 2),
    ] {
        writeln!(out, "{key:<40}   I     {value:12}").unwrap();
    }
    for (key, kind, values) in [
        ("Atomic numbers", "I", "1 1"),
        ("Nuclear charges", "R", "1 1"),
        (
            "Current cartesian coordinates",
            "R",
            "0.29 -0.17 0.31 1.18 0.36 -0.21",
        ),
        ("Shell types", "I", "0 0"),
        ("Number of primitives per shell", "I", "1 1"),
        ("Shell to atom map", "I", "1 2"),
        ("Primitive exponents", "R", "0.64 0.91"),
        ("Contraction coefficients", "R", "1 1"),
        ("Alpha MO coefficients", "R", "1 0"),
        ("Total SCF Density", "R", "0.98 0.42 0.18"),
    ] {
        writeln!(
            out,
            "{key:<40}   {kind}   N={:12}\n{values}",
            values.split_whitespace().count()
        )
        .unwrap();
    }
    out
}

#[test]
fn legacy_wfn_and_packed_fchk() {
    let weights = EXPONENTS.map(|e| (2.0 * e / std::f64::consts::PI).powf(0.75));
    let a = 0.7 * weights[0];
    let b = 0.3 * weights[1];
    let wfn = format!("Asymmetric H2\n GAUSSIAN 1 MOL ORBITALS 2 PRIMITIVES 2 NUCLEI\n H 1 (CENTRE 1) 0.29 -0.17 0.31 CHARGE = 1.0\n H 2 (CENTRE 2) 1.18 0.36 -0.21 CHARGE = 1.0\nCENTRE ASSIGNMENTS    1  2\nTYPE ASSIGNMENTS      1  1\nEXPONENTS 0.64 0.91\nMO 1 OCC NO = 2.0 ORB. ENERGY = -0.5\n{a:16.8E}{b:16.8E}\nEND DATA\n");
    let wfn = read_wfn(wfn.as_bytes()).unwrap();
    let fchk = read_fchk(fchk_atom_data().as_bytes()).unwrap();
    let oracle = |at| {
        2.0 * (a * gaussian(at, CENTERS[0], EXPONENTS[0])
            + b * gaussian(at, CENTERS[1], EXPONENTS[1]))
        .powi(2)
    };
    fitting(&fchk, "FCHK packed SCF density", oracle);
    let max_err = components(wfn.density_at(POINT).unwrap())
        .into_iter()
        .zip(components(fchk.density_at(POINT).unwrap()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("H2 WFN/FCHK maximum component difference = {max_err:.9e}");
    assert!(max_err < 8.0e-9);
}

#[test]
fn distinguishes_ecp_valence_and_wfx_edf() {
    let valence = r"
<Number of Nuclei>1</Number of Nuclei>
<Number of Primitives>1</Number of Primitives>
<Number of Occupied Molecular Orbitals>1</Number of Occupied Molecular Orbitals>
<Atomic Numbers>3</Atomic Numbers>
<Nuclear Charges>1</Nuclear Charges>
<Number of Core Electrons>2</Number of Core Electrons>
<Nuclear Cartesian Coordinates>0.29 -0.17 0.31</Nuclear Cartesian Coordinates>
<Primitive Centers>1</Primitive Centers>
<Primitive Types>1</Primitive Types>
<Primitive Exponents>0.64</Primitive Exponents>
<Molecular Orbital Occupation Numbers>1</Molecular Orbital Occupation Numbers>
<Molecular Orbital Primitive Coefficients>
<MO Number>1</MO Number>
0.7
</Molecular Orbital Primitive Coefficients>
";
    let expansion = r"
<Additional Electron Density Function (EDF)>
<Number of EDF Primitives>1</Number of EDF Primitives>
<EDF Primitive Centers>1</EDF Primitive Centers>
<EDF Primitive Types>1</EDF Primitive Types>
<EDF Primitive Exponents>2.06</EDF Primitive Exponents>
<EDF Primitive Coefficients>0.94</EDF Primitive Coefficients>
</Additional Electron Density Function (EDF)>
";
    let uncovered = read_wfx(valence.as_bytes()).unwrap();
    let restored = read_wfx(format!("{valence}{expansion}").as_bytes()).unwrap();
    assert_eq!(uncovered.density_model(), DensityModel::ValenceOnly);
    assert_eq!(restored.density_model(), DensityModel::ReconstructedCore);
    assert_eq!(restored.electron_count(), 3.0);
    assert_eq!(restored.orbital_electron_count(), 1.0);
    fitting(&restored, "WFX orbitals + EDF", |at| {
        0.49 * gaussian(at, CENTERS[0], 1.28) + 0.94 * gaussian(at, CENTERS[0], 2.06)
    });
    let lhs = restored.density_at(POINT).unwrap().density;
    let rhs = uncovered.density_at(POINT).unwrap().density;
    println!("Li valence = {rhs:.12e}, reconstructed = {lhs:.12e}");
    assert!((lhs - rhs - 0.94 * gaussian(POINT, CENTERS[0], 2.06)).abs() < 2.0e-16);
}

fn chain(range: i32) -> Wavefunction {
    let data = format!(
        r"NCIPLOT-WAVEFUNCTION 1
units bohr
atoms 2
1 0.29 -0.17 0.31 1
1 1.18 0.36 -0.21 1
periodic 1 0 0
lattice 2.4 0.31 -0.18
lattice 0.01 4.6 -0.02
lattice 0.08 -0.1 5.3
images {range} 0 0
primitives 2
1 0 0 0 0.64
2 0 0 0 0.91
core 0
kpoints 2
k 0.23 0 0 0.4 1
orbital 0.8 2
0.67 -0.12
0.16 0.41
-0.32 0.26
0.3 -0.11
k -0.37 0 0 0.6 1
orbital 1.4 1
0.68 -0.32
0.19 0.56
"
    );
    read_expanded(data.as_bytes()).unwrap()
}

fn chain_oracle(at: [f64; 3], range: i32) -> f64 {
    let value = |k: f64, coeffs: [[f64; 2]; 2]| {
        let mut orbital = [0.0; 2];
        for n in -range..=range {
            let phase = std::f64::consts::TAU * k * f64::from(n);
            for (a, c) in coeffs.into_iter().enumerate() {
                let center = std::array::from_fn(|i| CENTERS[a][i] + f64::from(n) * WIDTH[i]);
                let g = gaussian(at, center, EXPONENTS[a]);
                orbital[0] += g * (c[0] * phase.cos() - c[1] * phase.sin());
                orbital[1] += g * (c[0] * phase.sin() + c[1] * phase.cos());
            }
        }
        orbital[0].powi(2) + orbital[1].powi(2)
    };
    0.4 * 0.8
        * (value(0.23, [[0.67, -0.12], [0.16, 0.41]]) + value(0.23, [[-0.32, 0.26], [0.3, -0.11]]))
        + 0.6 * 1.4 * value(-0.37, [[0.68, -0.32], [0.19, 0.56]])
}

#[test]
fn non_gamma_complex_spinors_and_finite_image_seam() {
    let wf = chain(4);
    assert!((wf.electron_count() - 1.16).abs() < 1.0e-15);
    assert_eq!(
        wf.periodic_cell().unwrap().periodic_axes,
        [true, false, false]
    );
    fitting(&wf, "non-Gamma 1D Bloch spinor", |at| chain_oracle(at, 4));
    let original = components(wf.density_at(POINT).unwrap());
    let translated = components(
        wf.density_at(std::array::from_fn(|i| POINT[i] + WIDTH[i]))
            .unwrap(),
    );
    let invariant_error = original
        .into_iter()
        .zip(translated)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(invariant_error < 4.0e-15);
    let b = [0.01, 4.6, -0.02];
    let c = [0.08, -0.1, 5.3];
    let face = std::array::from_fn(|axis| b[axis] * 0.081 + c[axis] * 0.038);
    let image = std::array::from_fn(|i| face[i] + WIDTH[i]);
    let coarse_seam = (chain_oracle(face, 1) - chain_oracle(image, 1)).abs();
    let fine_seam = (chain_oracle(face, 4) - chain_oracle(image, 4)).abs();
    println!("Non-Gamma spatial translation max = {invariant_error:.9e}, raw face density mismatch {coarse_seam:.9e} to {fine_seam:.9e} e/Bohr^3");
    assert!(coarse_seam > 1.0e-5 && fine_seam < 1.0e-14);
    let samples = wf
        .sample_grid(wf.bounding_grid([5, 4, 3], 5.0).unwrap(), None, None)
        .unwrap();
    let data = analyze(&samples, Method::Nci).unwrap();
    assert_eq!(data.points.len(), 60);
    assert!(analyze(&samples, Method::Igmh).is_err());
}
