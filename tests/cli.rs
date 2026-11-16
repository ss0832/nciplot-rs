// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use std::{fs, path::PathBuf, process::Command};

struct Temporary(PathBuf);

impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn wfx(nuclei: &str, positions: &str, centers: &str, amplitudes: &str) -> String {
    let atoms = nuclei.split_whitespace().count();
    let exponents = "0.80 ".repeat(atoms);
    let types = "1 ".repeat(atoms);
    format!(
        "<Number of Nuclei>{atoms}</Number of Nuclei>
         <Number of Primitives>{atoms}</Number of Primitives>
         <Number of Occupied Molecular Orbitals>1</Number of Occupied Molecular Orbitals>
         <Atomic Numbers>{nuclei}</Atomic Numbers>
         <Nuclear Charges>{nuclei}</Nuclear Charges>
         <Nuclear Cartesian Coordinates>{positions}</Nuclear Cartesian Coordinates>
         <Primitive Centers>{centers}</Primitive Centers>
         <Primitive Types>{types}</Primitive Types>
         <Primitive Exponents>{exponents}</Primitive Exponents>
         <Molecular Orbital Occupation Numbers>{atoms}</Molecular Orbital Occupation Numbers>
         <Molecular Orbital Primitive Coefficients>
         <MO Number>1</MO Number>
         {amplitudes}
         </Molecular Orbital Primitive Coefficients>"
    )
}

#[test]
fn cli_analyzes_wavefunctions_and_writes_a_single_offline_view() {
    let binary = env!("CARGO_BIN_EXE_nciplot-rs");
    let dir =
        Temporary(std::env::temp_dir().join(format!("nciplot-rs-cli-{}", std::process::id())));
    fs::create_dir_all(&dir.0).unwrap();
    let normalization = (1.6 / std::f64::consts::PI).powf(0.75);
    let bond_norm = (2.0
        * (1.0 + (-0.4_f64 * (1.37_f64.powi(2) + 0.1_f64.powi(2) + 0.24_f64.powi(2))).exp()))
    .sqrt();
    let lo = normalization / bond_norm;
    fs::write(
        dir.0.join("dimer.wfx"),
        wfx(
            "1 1",
            "-0.68 0.12 0.08 0.69 0.02 -0.16",
            "1 2",
            &format!("{lo} {lo}"),
        ),
    )
    .unwrap();
    fs::write(
        dir.0.join("atom.wfx"),
        wfx("1", "0.31 -0.34 0.22", "1", &normalization.to_string()),
    )
    .unwrap();
    let version = Command::new(binary).arg("--version").output().unwrap();
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).starts_with("nciplot-rs 0.1.0"));
    for method in ["nci", "igm", "igmh", "igmh-hirshfeld"] {
        let mut cmd = Command::new(binary);
        cmd.current_dir(&dir.0).args([
            method,
            "dimer.wfx",
            "--prefix",
            method,
            "--shape",
            "7,6,5",
            "--padding",
            "2.4",
            "--fragments",
            "0,8",
            "--surface-density",
            "signed",
            "--density-min",
            "-0.025",
            "--density-max",
            "-0.004",
            "--color-min",
            "-0.028",
            "--color-max",
            "-0.0025",
        ]);
        if method != "nci" {
            cmd.args(["--proatom", "1=atom.wfx"]);
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            stdout
                .lines()
                .filter(|name| name.ends_with(".html"))
                .count(),
            1
        );
        assert_eq!(stdout.lines().count(), 5);
        let html = fs::read_to_string(dir.0.join(format!("{method}.html"))).unwrap();
        assert!(html.contains("\"colorMin\":-0.028,\"colorMax\":-0.0025"));
        assert!(html.contains("\"surfaceDensityMin\":-0.025,\"surfaceDensityMax\":-0.004,\"signedSurfaceDensity\":true"));
        let csv = fs::read_to_string(dir.0.join(format!("{method}.csv"))).unwrap();
        assert_eq!(csv.lines().count(), 211);
    }
    let invalid = Command::new(binary)
        .current_dir(&dir.0)
        .args(["igmh", "dimer.wfx", "--shape", "2,2,2"])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("neutral proatom samples"));
    let invalid = Command::new(binary)
        .current_dir(&dir.0)
        .args([
            "nci",
            "dimer.wfx",
            "--shape",
            "2,2,2",
            "--density-min",
            "-0.01",
        ])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("nonnegative bounds"));
}
