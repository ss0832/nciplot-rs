# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Verify the installed native API and command-line driver."""

import csv
import math
import os
import subprocess
import sys
import tempfile
import unittest
from importlib.metadata import metadata
from pathlib import Path

import nciplot_rs as nci


def wfx(numbers, positions, amplitudes, occupation):
    count = len(numbers)
    def join(values):
        return " ".join(map(str, values))
    return f"""
<Number of Nuclei>{count}</Number of Nuclei>
<Number of Primitives>{count}</Number of Primitives>
<Number of Occupied Molecular Orbitals>1</Number of Occupied Molecular Orbitals>
<Atomic Numbers>{join(numbers)}</Atomic Numbers>
<Nuclear Charges>{join(numbers)}</Nuclear Charges>
<Nuclear Cartesian Coordinates>{join(positions)}</Nuclear Cartesian Coordinates>
<Primitive Centers>{join(range(1, count + 1))}</Primitive Centers>
<Primitive Types>{join([1] * count)}</Primitive Types>
<Primitive Exponents>{join([0.8] * count)}</Primitive Exponents>
<Molecular Orbital Occupation Numbers>{occupation}</Molecular Orbital Occupation Numbers>
<Molecular Orbital Primitive Coefficients>
<MO Number>1</MO Number>
{join(amplitudes)}
</Molecular Orbital Primitive Coefficients>
"""


class InstalledTests(unittest.TestCase):
    def test_non_gamma_periodic_export(self):
        cli = Path(sys.executable).with_name("nciplot-rs.exe" if sys.platform == "win32" else "nciplot-rs")
        with tempfile.TemporaryDirectory() as working:
            path = Path(working)
            source = path / "chain.ncwf"
            source.write_text("""NCIPLOT-WAVEFUNCTION 1
units bohr
atoms 1
1 0.29 -0.17 0.31 1
periodic 1 0 0
lattice 2.4 0.31 -0.18
lattice 0.01 4.6 -0.02
lattice 0.08 -0.10 5.3
images 4 0 0
primitives 1
1 0 0 0 0.64
core 0
kpoints 1
k 0.23 0 0 1.0 1
orbital 0.8 2
0.67 -0.12
-0.32 0.26
""", encoding="utf-8")
            wavefunction = nci.load_wavefunction(source)
            self.assertEqual(wavefunction.density_model, "all-electron")
            self.assertAlmostEqual(wavefunction.electron_count, 0.8, places=15)
            self.assertEqual(tuple(wavefunction.periodic_cell["periodic_axes"]), (True, False, False))
            reference = wavefunction.density_at([0.61, 0.47, 0.91])
            moved = wavefunction.density_at([3.01, 0.78, 0.73])
            self.assertAlmostEqual(reference[0], moved[0], places=15)
            for left, right in zip(reference[1] + reference[2], moved[1] + moved[2]):
                self.assertAlmostEqual(left, right, places=14)
            result = wavefunction.sample(shape=(5, 4, 3)).analyze("nci")
            self.assertEqual(tuple(result.report()["periodic_cell"]["image_ranges"]), (4, 0, 0))
            result.write_csv(path / "native.csv")
            args = [cli, "nci", source, "--shape", "5,4,3", "--prefix", path / "cli"]
            subprocess.run(args, check=True, capture_output=True)
            self.assertEqual((path / "native.csv").read_bytes(), (path / "cli.csv").read_bytes())
            failure = subprocess.run([cli, "igmh", source, "--shape", "5,4,3"], check=False, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("require neutral proatom", failure.stderr)

    def test_metadata(self):
        package = metadata("nciplot-rs-python")
        self.assertEqual(package["Version"], nci.__version__)
        self.assertEqual(package["License-Expression"], "Apache-2.0")
        cli = Path(sys.executable).with_name("nciplot-rs.exe" if sys.platform == "win32" else "nciplot-rs")
        self.assertEqual(subprocess.check_output([cli, "--version"], text=True).strip(),
                         "nciplot-rs " + nci.__version__)

    def test_analytic_density_molden_and_wfx(self):
        normalizer = (1.6 / math.pi) ** 0.75
        position = [0.11, -0.26, 0.07]
        with tempfile.TemporaryDirectory() as working:
            directory = Path(working)
            atom = directory / "h.wfx"
            atom.write_text(wfx([1], position, [normalizer], 1), encoding="utf-8")
            molden = directory / "h.molden.input"
            molden.write_text("""[Molden Format]
[Atoms] AU
H 1 1 0.11 -0.26 0.07
[GTO]
1 0
s 1 1.0
0.8 1.0
[MO]
Spin= Alpha
Occup= 1
1 1
""", encoding="utf-8")
            for path in (atom, molden):
                wavefunction = nci.load_wavefunction(path)
                density, gradient, hessian = wavefunction.density_at(position)
                self.assertAlmostEqual(wavefunction.electron_count, 1.0, places=15)
                self.assertAlmostEqual(density, normalizer**2, places=15)
                self.assertEqual(tuple(gradient), (0.0, 0.0, 0.0))
                for value in (hessian[0], hessian[3], hessian[5]):
                    self.assertAlmostEqual(value, -3.2 * density, places=15)
                self.assertEqual(hessian[1], 0.0)

    def test_native_and_command_line_fields_and_products(self):
        binary = Path(sys.executable).with_name("nciplot-rs.exe" if sys.platform == "win32" else "nciplot-rs")
        shape = [7, 6, 5]
        distance2 = 1.37**2 + 0.1**2 + 0.24**2
        normalizer = (1.6 / math.pi) ** 0.75
        amplitude = normalizer / math.sqrt(2.0 * (1.0 + math.exp(-0.4 * distance2)))
        with tempfile.TemporaryDirectory() as working:
            path = Path(working)
            atoms = path / "h2.wfx"
            atoms.write_text(
                wfx([1, 1], [-0.68, 0.12, 0.08, 0.69, 0.02, -0.16], [amplitude] * 2, 2),
                encoding="utf-8",
            )
            atom = path / "h.wfx"
            atom.write_text(wfx([1], [0, 0, 0], [normalizer], 1), encoding="utf-8")
            source = nci.load_wavefunction(atoms)
            dataset = source.sample(shape=shape, padding=2.4, fragments=[1, 8], proatoms={1: atom})
            for method in ("nci", "igm", "igmh", "igmh-hirshfeld"):
                with self.subTest(method=method):
                    options = nci.ViewerOptions(density="signed", density_range=(-0.025, -0.004),
                                                color_range=(-0.028, 0.018))
                    analysis = dataset.analyze(method)
                    analysis.write_csv(path / "py.csv")
                    analysis.write_html(path / "py.html", options=options)
                    analysis.write_cube(path / "py.cube", "signed-density")
                    args = [binary, method, atoms, "--shape", "7,6,5", "--padding", "2.4",
                            "--fragments", "1,8", "--surface-density", "signed",
                            "--density-min", "-0.025", "--density-max", "-0.004",
                            "--color-min", "-0.028", "--color-max", "0.018",
                            "--prefix", path / "cli"]
                    if method != "nci":
                        args.extend(["--proatom", f"1={atom}"])
                    cli = subprocess.run(args, check=True, capture_output=True, text=True)
                    self.assertIn("210 nodes", cli.stderr)
                    for left, right in (("csv", "csv"), ("html", "html"), ("cube", "signed-density.cube")):
                        self.assertEqual((path / f"py.{left}").read_bytes(), (path / f"cli.{right}").read_bytes())
                    if rust_cli := os.environ.get("NCIPLOT_RUST_CLI"):
                        args[0] = rust_cli
                        args[args.index("--prefix") + 1] = path / "rust"
                        subprocess.run(args, check=True, capture_output=True)
                        for suffix in ("csv", "html", "field.cube", "signed-density.cube", "density.cube"):
                            self.assertEqual((path / f"rust.{suffix}").read_bytes(), (path / f"cli.{suffix}").read_bytes())
                    report = analysis.report()
                    self.assertEqual(tuple(report["shape"]), tuple(shape))
                    self.assertEqual(report["units"]["hessian"], "e/bohr^5")
                    with (path / "cli.csv").open(encoding="utf-8", newline="") as raw:
                        rows = list(csv.DictReader(raw))
                    for index, row in enumerate(rows):
                        for key in ("density", "signed_density", "rdg"):
                            column = key if key == "rdg" else f"{key}_e_bohr3"
                            self.assertAlmostEqual(report[key][index], float(row[column]), places=14)
                        for values, suffix, prefix, components in (
                            (report["gradient"], "4", "grad", ("x", "y", "z")),
                            (report["hessian"], "5", "hess", ("xx", "xy", "xz", "yy", "yz", "zz")),
                        ):
                            for component, value in zip(components, values[index]):
                                self.assertAlmostEqual(value, float(row[f"{prefix}_{component}_e_bohr{suffix}"]), places=14)
            with self.assertRaises(ValueError):
                dataset.analyze("unknown")
            with self.assertRaises(FileNotFoundError):
                nci.load_wavefunction(path / "missing.wfx")


if __name__ == "__main__":
    unittest.main()
