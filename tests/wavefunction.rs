// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use nciplot_rs::{read_molden, read_wfx, Wavefunction};

const WFX: &str = r"
<Number of Nuclei> 2 </Number of Nuclei>
<Number of Primitives> 4 </Number of Primitives>
<Number of Occupied Molecular Orbitals> 2 </Number of Occupied Molecular Orbitals>
<Number of Perturbations> 0 </Number of Perturbations>
<Atomic Numbers> 8 1 </Atomic Numbers>
<Nuclear Charges> 8 1 </Nuclear Charges>
<Nuclear Cartesian Coordinates>
 0.18 -0.23 0.31   -0.42 0.13 0.57
</Nuclear Cartesian Coordinates>
<Primitive Centers> 1 1 2 2 </Primitive Centers>
<Primitive Types> 1 2 10 4 </Primitive Types>
<Primitive Exponents> 5.1D-1 0.81 1.17 0.63 </Primitive Exponents>
<Molecular Orbital Occupation Numbers> 1.66 0.37 </Molecular Orbital Occupation Numbers>
<Molecular Orbital Primitive Coefficients>
<MO Number> 1 </MO Number>
0.83 -0.46 0.32 -0.17
<MO Number> 2 </MO Number>
-0.31 0.51 0.65 -0.24
</Molecular Orbital Primitive Coefficients>
";

const MOLDEN: &str = r"
[Molden Format]
[Atoms] AU
O 1 8 0.18 -0.23 0.31
[GTO]
1 0
s 1 1.00
 0.63 1.0
d 1 1.00
 0.90 1.0
[5D7F]
[MO]
 Sym= A
 Ene= -0.7
 Spin= Alpha
 Occup= 1.4
 1 0.64
 2 -0.32
";

const AT: [f64; 3] = [0.33, 0.41, -0.16];
const WFX_REFERENCE: [f64; 9] = [
    -0.779_844_712_560_802_2,
    -0.820_480_587_347_421_7,
    0.451_303_949_381_334_1,
    -0.305_399_804_924_453,
    1.389_282_774_645_280_3,
    -0.745_241_114_316_678_9,
    -0.021_649_369_725_769_172,
    -0.623_924_664_745_225_8,
    -0.885_099_058_451_291_2,
];
const MOLDEN_REFERENCE: [f64; 9] = [
    -0.002_129_975_986_684_557,
    -0.009_087_897_543_187_444,
    0.210_919_453_841_526_36,
    -0.021_403_400_043_743_206,
    -0.030_735_189_898_721_392,
    -0.036_013_220_504_231_51,
    -0.145_336_650_145_775,
    -0.153_656_407_484_721_12,
    0.177_322_199_901_069_84,
];
const COMPONENTS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

fn wfx_oracle([x, y, z]: [f64; 3]) -> f64 {
    let r1 = (x - 0.18).powi(2) + (y + 0.23).powi(2) + (z - 0.31).powi(2);
    let r2 = (x + 0.42).powi(2) + (y - 0.13).powi(2) + (z - 0.57).powi(2);
    let a = (-0.51 * r1).exp();
    let b = (x - 0.18) * (-0.81 * r1).exp();
    let c = (y - 0.13) * (z - 0.57) * (-1.17 * r2).exp();
    let d = (z - 0.57) * (-0.63 * r2).exp();
    1.66 * (0.83 * a - 0.46 * b + 0.32 * c - 0.17 * d).powi(2)
        + 0.37 * (-0.31 * a + 0.51 * b + 0.65 * c - 0.24 * d).powi(2)
}

fn molden_oracle([x, y, z]: [f64; 3]) -> f64 {
    let (x, y, z) = (x - 0.18, y + 0.23, z - 0.31);
    let r2 = x * x + y * y + z * z;
    let pi = std::f64::consts::PI;
    let s = (1.26 / pi).powf(0.75) * (-0.63 * r2).exp();
    let d_norm = 1.8_f64.powf(1.75) / (0.75 * pi.powf(1.5)).sqrt();
    let d = d_norm * (z * z - (x * x + y * y) / 2.0) * (-0.9 * r2).exp();
    1.4 * (0.64 * s - 0.32 * d).powi(2)
}

fn verify_all_derivatives(
    wavefunction: &Wavefunction,
    oracle: impl Fn([f64; 3]) -> f64,
    format: &str,
) {
    let production = wavefunction.density_at(AT).unwrap();
    let exact_density = oracle(AT);
    // Independent 84-digit density stencils at h=1e-20 Bohr give a
    // reference for the absolute derivative error as well as the fit below.
    let reference = if format == "WFX" {
        WFX_REFERENCE
    } else {
        MOLDEN_REFERENCE
    };
    let absolute_error = production
        .gradient
        .into_iter()
        .chain(production.hessian)
        .zip(reference)
        .map(|(calculated, expected)| (calculated - expected).abs())
        .fold(0.0, f64::max);
    println!(
        "{format} maximum absolute error against high-precision reference {absolute_error:.8e}"
    );
    assert!(absolute_error < 2.0e-15);
    assert!((production.density - exact_density).abs() < 8.0e-16);
    let steps = [0.028_f64, 0.018, 0.011, 0.007];
    let mut errors = [[0.0; 4]; 9];
    for (n, &h) in steps.iter().enumerate() {
        let f = |i: usize, di: f64, j: usize, dj: f64| {
            let mut point = AT;
            point[i] += di;
            point[j] += dj;
            oracle(point)
        };
        for (i, derivative) in production.gradient.into_iter().enumerate() {
            let expected = (f(i, h, i, 0.0) - f(i, -h, i, 0.0)) / (2.0 * h);
            errors[i][n] = (expected - derivative).abs();
        }
        for (component, (i, j)) in COMPONENTS.into_iter().enumerate() {
            let expected = if i == j {
                (f(i, h, i, 0.0) - 2.0 * exact_density + f(i, -h, i, 0.0)) / h.powi(2)
            } else {
                (f(i, h, j, h) - f(i, h, j, -h) - f(i, -h, j, h) + f(i, -h, j, -h)) / (4.0 * h * h)
            };
            errors[component + 3][n] = (expected - production.hessian[component]).abs();
        }
    }
    let x = steps.map(f64::ln);
    let avg = |values: [f64; 4]| values.into_iter().sum::<f64>() / 4.0;
    let dx = x.map(|value| value - avg(x));
    let ssx: f64 = dx.into_iter().map(|x| x * x).sum();
    for (component, err) in errors.into_iter().enumerate() {
        assert!(err.into_iter().all(|error| error > 0.0));
        let y = err.map(f64::ln);
        let dy = y.map(|value| value - avg(y));
        let slope = dx.into_iter().zip(dy).map(|(x, y)| x * y).sum::<f64>() / ssx;
        let intercept = avg(y) - slope * avg(x);
        let ssr: f64 = dx
            .into_iter()
            .zip(dy)
            .map(|(x, y)| (y - slope * x).powi(2))
            .sum();
        let ssy: f64 = dy.into_iter().map(|y| y * y).sum();
        let fit = 1.0 - ssr / ssy;
        println!("{format} component {component}: p={slope:.8} R²={fit:.10} C={:.8e} h=[0.007,0.028] abs_error={err:?}", intercept.exp());
        assert!((slope - 2.0).abs() < 0.02 && fit > 0.99995);
    }
}

#[test]
fn wfx_density_and_all_cartesian_derivatives() {
    let wfx = read_wfx(WFX.as_bytes()).unwrap();
    assert!((wfx.electron_count() - 2.03).abs() < 1.0e-15);
    verify_all_derivatives(&wfx, wfx_oracle, "WFX");
}

#[test]
fn molden_spherical_density_and_all_cartesian_derivatives() {
    let molden = read_molden(MOLDEN.as_bytes()).unwrap();
    verify_all_derivatives(&molden, molden_oracle, "Molden");
}

#[test]
fn incompatible_wavefunction_features_return_errors() {
    let ecp = WFX.replace("<Nuclear Charges> 8 1", "<Nuclear Charges> 6.5 1");
    assert!(read_wfx(ecp.as_bytes())
        .unwrap_err()
        .to_string()
        .contains("effective nuclear"));
    let no_primitive = WFX.replace("1 2 10 4", "1 2 0 4");
    assert!(read_wfx(no_primitive.as_bytes())
        .unwrap_err()
        .to_string()
        .contains("primitive type"));
    let complex = MOLDEN.replace(" 1 0.64", " 1 0.64 0.1");
    assert!(read_molden(complex.as_bytes())
        .unwrap_err()
        .to_string()
        .contains("real value"));
    let element = MOLDEN.replace("O 1 8", "O 2 8");
    assert!(read_molden(element.as_bytes())
        .unwrap_err()
        .to_string()
        .contains("atom indices"));
}
