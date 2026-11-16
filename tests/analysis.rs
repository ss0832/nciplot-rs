// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use nciplot_rs::{
    analyze, read_dataset, write_csv, write_cube, write_html, Atom, CubeField, Dataset,
    DensityKind, DensitySample, Descriptor, Grid, Method, ProatomSample, Units, ViewerOptions,
};

const ASYMMETRIC: &str = r"
NCIPLOT-RS 1
units bohr
shape 1 1 1
origin -0.15 0.27 0.19
step 0.31 0.05 0.07
step 0.02 0.23 -0.01
step 0.08 -0.06 0.29
atoms 3
8 -0.05 -0.11 0.02 10
1 -0.63 0.33 1.01 10
1 1.43 0.60 0.41 25
samples molecular with-proatoms
0.4 0.14 -0.12 0.19 0.5 -1.5 0 -1.5 0 -1
1 0.4 -0.3 0.6
2 -0.8 0.1 -0.2
1 0.15 0.5 0.2
";

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.17e} versus {expected:.17e}"
    );
}

#[test]
fn stockholder_and_nci_closed_forms() {
    let dataset = read_dataset(ASYMMETRIC.as_bytes()).unwrap();
    let norm = |v: [f64; 3]| v.into_iter().map(|x| x * x).sum::<f64>().sqrt();
    let reference = [
        (
            Method::Igm,
            [[0.4, -0.3, 0.6], [-0.8, 0.1, -0.2], [0.15, 0.5, 0.2]],
            [-0.40, -0.20, 0.40],
        ),
        (
            Method::Igmh,
            [
                [-0.01125, 0.0075, 0.0025],
                [0.1375, -0.055, 0.145],
                [0.01375, -0.0725, 0.0425],
            ],
            [0.12625, -0.0475, 0.1475],
        ),
        (
            Method::IgmhHirshfeld,
            [
                [0.08125, -0.0675, 0.0925],
                [0.0025, -0.065, 0.045],
                [0.05625, 0.0125, 0.0525],
            ],
            [0.08375, -0.1325, 0.1375],
        ),
    ];
    for (method, vectors, pair) in reference {
        let data = analyze(&dataset, method).unwrap();
        let point = data.points[0];
        // The x-y block brackets the uncoupled z eigenvalue -1.
        close(point.lambda2, -1.0, 5.0e-16);
        close(point.signed_density, -0.4, 0.0);
        let rdg = norm([0.14, -0.12, 0.19])
            / (2.0 * (3.0 * std::f64::consts::PI.powi(2)).cbrt() * 0.4_f64.powf(4.0 / 3.0));
        close(point.rdg.unwrap(), rdg, 1.0e-16);
        let sum = std::array::from_fn(|axis| pair[axis] + vectors[2][axis]);
        let inter = norm(pair) + norm(vectors[2]) - norm(sum);
        let intra = norm(vectors[0]) + norm(vectors[1]) - norm(pair);
        let delta = point.delta_g.unwrap();
        close(delta.inter, inter, 4.0e-16);
        close(delta.intra, intra, 4.0e-16);
        close(delta.total, inter + intra, 4.0e-16);
        if method != Method::Igm {
            assert!(data.density_residual.unwrap() < 1.0e-16);
            assert!(data.gradient_residual.unwrap() < 1.0e-16);
        }
    }
}

#[test]
fn unit_conversion_and_density_semantics() {
    let density = DensitySample {
        density: 0.4,
        gradient: [0.5, 0.6, -0.7],
        hessian: [-0.2, 0.12, -0.07, -0.16, 0.06, 0.36],
    };
    let reference = ProatomSample {
        density: 0.24,
        gradient: [-0.1, 0.04, 0.07],
    };
    let origin = [-0.3, 0.2, 0.7];
    let steps = [[0.1, 0.03, 0.01], [-0.02, 0.2, 0.013], [0.01, -0.05, 0.25]];
    let geometry = [0.3, -0.2, -0.03];
    let datasets =
        [(Units::Bohr, 1.0_f64), (Units::Angstrom, 0.529_177_210_903)].map(|(units, s)| {
            Dataset::new(
                Grid::new(
                    [2, 1, 1],
                    origin.map(|value| value * s),
                    steps.map(|step| step.map(|value| value * s)),
                    units,
                )
                .unwrap(),
                vec![Atom {
                    atomic_number: 1,
                    position: geometry.map(|value| value * s),
                    fragment: 3,
                }],
                DensityKind::Molecular,
                vec![
                    DensitySample {
                        density: density.density / s.powi(3),
                        gradient: density.gradient.map(|value| value / s.powi(4)),
                        hessian: density.hessian.map(|value| value / s.powi(5)),
                    };
                    2
                ],
                Some(vec![
                    ProatomSample {
                        density: reference.density / s.powi(3),
                        gradient: reference.gradient.map(|value| value / s.powi(4)),
                    };
                    2
                ]),
                units,
            )
            .unwrap()
        });
    let [a, b] = datasets
        .each_ref()
        .map(|d| analyze(d, Method::IgmhHirshfeld).unwrap().points[0]);
    for (a, b) in [
        (a.density, b.density),
        (a.lambda2, b.lambda2),
        (a.signed_density, b.signed_density),
        (a.rdg.unwrap(), b.rdg.unwrap()),
    ] {
        close(a, b, 3.0e-16);
    }
    assert_eq!(a.delta_g.unwrap().total, 0.0);
    assert_eq!(b.delta_g.unwrap().total, 0.0);
    let mut cube = Vec::new();
    write_cube(
        &mut cube,
        &datasets[0],
        &analyze(&datasets[0], Method::Nci).unwrap(),
        CubeField::SignedDensity,
    )
    .unwrap();
    let cube = std::str::from_utf8(&cube).unwrap();
    let lines: Vec<_> = cube.lines().collect();
    assert!(lines[2].starts_with("1 "));
    assert!(lines[3].starts_with("2 "));
    assert_eq!(lines.len(), 9);
    assert!(lines[7].trim_start().starts_with('-'));
}

#[test]
fn vacuum_keeps_the_reduced_gradient_undefined() {
    let grid = Grid::new(
        [1; 3],
        [0.0; 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        Units::Bohr,
    )
    .unwrap();
    let atom = Atom {
        atomic_number: 1,
        position: [0.0; 3],
        fragment: 0,
    };
    let vacuum = DensitySample {
        density: 0.0,
        gradient: [0.0; 3],
        hessian: [0.0; 6],
    };
    let dataset = Dataset::new(
        grid.clone(),
        vec![atom],
        DensityKind::Molecular,
        vec![vacuum],
        None,
        Units::Bohr,
    )
    .unwrap();
    let result = analyze(&dataset, Method::Nci).unwrap();
    assert_eq!(result.zero_density_points, 1);
    assert!(result.points[0].rdg.is_none());
    assert!(analyze(&dataset, Method::Igmh).is_err());
    let mut csv = Vec::new();
    write_csv(&mut csv, &dataset, &result).unwrap();
    assert!(std::str::from_utf8(&csv).unwrap().ends_with(",,,,\n"));
    let inconsistent = Dataset::new(
        grid,
        vec![atom],
        DensityKind::Molecular,
        vec![DensitySample {
            density: 0.1,
            ..vacuum
        }],
        Some(vec![ProatomSample {
            density: 0.0,
            gradient: [0.0; 3],
        }]),
        Units::Bohr,
    )
    .unwrap();
    assert!(analyze(&inconsistent, Method::Igmh)
        .unwrap_err()
        .to_string()
        .contains("zero promolecule"));
}

fn packed_len(html: &str, key: &str) -> usize {
    let prefix = format!("\"{key}\":\"");
    let (_, data) = html.split_once(&prefix).unwrap();
    let (data, _) = data.split_once('"').unwrap();
    data.len() / 4 * 3
        - data
            .chars()
            .rev()
            .take_while(|&symbol| symbol == '=')
            .count()
}

#[test]
fn indexed_viewer_keeps_all_affine_surface_and_scatter_data() {
    let grid = Grid::new(
        [2, 2, 3],
        [-0.3, 0.1, -0.07],
        [[0.31, 0.05, 0.07], [0.02, 0.23, -0.01], [0.08, -0.06, 0.29]],
        Units::Bohr,
    )
    .unwrap();
    let atom = Atom {
        atomic_number: 8,
        position: [0.0; 3],
        fragment: 0,
    };
    let samples: Vec<_> = (0..grid.point_count())
        .map(|index| DensitySample {
            density: 0.011 + 0.001 * index as f64,
            gradient: [if index < 6 { 0.0 } else { 0.3 }, 0.0, 0.0],
            hessian: [-1.0, 0.0, 0.0, -0.3, 0.0, 0.6],
        })
        .collect();
    let dataset = Dataset::new(
        grid,
        vec![atom],
        DensityKind::Molecular,
        samples,
        None,
        Units::Bohr,
    )
    .unwrap();
    let analysis = analyze(&dataset, Method::Nci).unwrap();
    let mut html = Vec::new();
    let mut options = ViewerOptions::new(Descriptor::Rdg);
    options.surface_density_min = 0.013;
    options.surface_density_max = 0.018;
    write_html(&mut html, &dataset, &analysis, options).unwrap();
    let html = std::str::from_utf8(&html).unwrap();
    assert!(html.contains("\"surfaceDensityMin\":0.013,\"surfaceDensityMax\":0.018"));
    assert_eq!(packed_len(html, "scatter"), 12 * 3 * 8);
    assert!(packed_len(html, "faces") > 0);
    assert!(packed_len(html, "vertices") / 20 < packed_len(html, "faces") / 4);
    assert!(html.contains("nciRange.value = \"full\""));
    assert!(html.contains("function clipDensity("));
    assert!(!html.contains("<script src="));
}
