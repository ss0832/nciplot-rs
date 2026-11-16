# nciplot-rs

Rust library and command-line tool for isolated molecular NCI, promolecular
IGM, and IGMH. It accepts real Gaussian wavefunctions or a grid of analytic
real-space density derivatives. Results are available as CSV, CUBE, and a
single offline HTML document containing the rotatable molecule, interaction
isosurface, and scatter plots.

## Build and run

Requires Rust 1.85 or later. The default Rust library and executable use the
Rust standard library. The optional `python` feature builds the native bindings.

```sh
cargo install --path . --locked
nciplot-rs --version
nciplot-rs nci water-dimer.molden --shape 41,41,61 --prefix water-nci
nciplot-rs igmh water-dimer.wfx --fragments 0,0,0,1,1,1 --proatom 1=hydrogen.wfx --proatom 8=oxygen.wfx --prefix water-igmh
nciplot-rs nci water-dimer.molden --surface-density signed --density-min -0.025 --density-max -0.004 --color-min -0.028 --color-max 0.018 --prefix water-negative
```

`nci` uses the density alone. `igm`, `igmh`, and `igmh-hirshfeld` also use one
neutral free-atom reference per element for wavefunction input. Supply spherical
neutral atomic references or analytic spherical proatom data for the published
model. For an open-shell atomic wavefunction, a spherical ensemble may be
required. The program checks the atom and electron count of atomic
wavefunction references. The reference's electronic-structure model,
normalization and spherical averaging are input responsibilities.

The CLI builds a rectangular Bohr grid padded by 5 Bohr beyond the nuclei
unless changed by `--padding`. For direct affine grids, use the analytic
sample format or the Rust or Python API. For periodic wavefunctions, the
default grid spans the supplied lattice from zero to the opposite cell faces.
The CLI writes `.html`, `.csv`, `.field.cube`,
`.signed-density.cube`, and `.density.cube` at the chosen prefix. The default
field is RDG for NCI, interfragment Δg when there are multiple fragments,
and total Δg otherwise. Select `--field rdg|total|inter|intra` to override this.
`--help` lists scientific and visualization options.

## Python

Install the wheel or source archive with pip. From this checkout use
`python -m pip install .` to build locally with Rust and maturin.
The distribution is `nciplot-rs-python` and import name is `nciplot_rs`.
The supplied Python launcher runs the same driver as the Rust `nciplot-rs`
command. The module requires CPython 3.9 or newer.

```python
import nciplot_rs as nci

wavefunction = nci.load_wavefunction("water-dimer.molden")
dataset = wavefunction.sample(shape=(41, 41, 61), fragments=(0, 0, 0, 1, 1, 1))
result = dataset.analyze("nci")
report = result.report()
result.write_html(
    "water.html",
    options=nci.ViewerOptions(
        density="signed", density_range=(-0.025, -0.004),
        color_range=(-0.028, 0.018),
    ),
)
result.write_csv("water.csv")
result.write_cube("water-signed-density.cube", "signed-density")
```

Pass `proatoms={1: "hydrogen.wfx", 8: "oxygen.wfx"}` to `sample` to provide
neutral atomic references for `analyze("igm")` or `analyze("igmh")`.
`Grid(shape, origin, steps, units="bohr")` specifies an affine grid accepted by
`sample(grid=grid)`. `load_dataset(path)` accepts the analytic sample format.
`Wavefunction.density_at(point)` returns density, Cartesian gradient and
packed Hessian at a Bohr point. `Analysis.report()` includes flat fields in
third-axis-fastest order, grid metadata, units, closure residuals when
available, and the periodic cell and density model when known. Undefined
scalar fields appear as `None`. The wavefunction sampler and exporters
release the GIL during Rust computation and file I/O.

## Wavefunction import

| Format | Supported data |
| --- | --- |
| WFX (`.wfx`) | Real occupied orbitals, including fractional occupations, expressed in unnormalized Cartesian Gaussian primitives. Pure spherical functions are accepted through their Cartesian expansions. Supports ECP valence density and an explicit EDF for core reconstruction. Primitive types follow the standard through G and its prescribed order through angular momentum 12. |
| Molden (`.molden`, `.molden.input`) | Real all-electron `[Atoms]`, `[GTO]`, and `[MO]` in standard ordering and normalized contraction convention. Coordinates may use AU or Angs. Supports s, p, sp, and Cartesian or real spherical d, f, and g shells. Shell scale must be 1. |
| WFN (`.wfn`) | Real Gaussian s through f primitive expansions in the original fixed-width interchange convention. Effective charges distinguish ECP valence density. Higher angular momentum extensions vary by exporter. |
| FCHK (`.fchk`, `.fch`) | Real s/p/sp/d/f basis, Cartesian or spherical. Uses the packed `Total SCF Density` when present. For files without that matrix it uses the leading occupied alpha/beta MOs with integer SCF counts, requiring the standard occupied-first order. Explicit MO occupations without a density matrix are rejected. ECP information supplies valence density. |
| Expanded Gaussian (`.ncwf`) | Versioned text input for isolated complex orbitals and spinors, 1D to 3D Bloch orbitals with explicit weights and image ranges, and optional reconstructed core density. |

The import path evaluates the charge density and all nine independent
first/second derivatives analytically. WFX, WFN, Molden and FCHK accept
separate real alpha and beta orbitals. Ghost centers and nonzero WFX
perturbation counts are rejected. FCHK evaluates the SCF density and keeps
correlated density matrices outside its scope. ECP files without a
supplied core expansion yield a `valence-only` density. Reconstructed
density adds the supplied linear core Gaussian expansion to the valence
density and records `reconstructed-core`. Electron counts use orbital
occupations and the nominal replaced core population. The caller supplies
the normalization and accuracy of an EDF. Molden lacks effective nuclear
charges and must use the listed all-electron conventions. The CLI performs
no SCF calculation.

[WFX specification](https://aim.tkgristmill.com/wfxformat.html).
[Molden specification](https://www.theochem.ru.nl/molden/molden_format.html).
[WFN fixed-width records](https://www.quimica.urv.es/XAIM/Xaim-1.0.pdf).
[FCHK specification](https://tungsten.ch.uj.edu.pl/doc/g09doc/g09ur/f_formchk.htm).

### Expanded Gaussian interchange

Exporters can generate the following plain-text schema. It is specific to
this crate and does not extend any of the four conventional formats.

```text
NCIPLOT-WAVEFUNCTION 1
units bohr
atoms 1
1 0.31 -0.17 0.29 1
periodic 1 0 0
lattice 2.4 0.31 -0.18
lattice 0.01 4.60 -0.02
lattice 0.08 -0.10 5.30
images 4 0 0
primitives 1
1 0 0 0 0.64
core 0
kpoints 1
k 0.23 0 0 1.0 1
orbital 0.8 2
0.67 -0.12
-0.32 0.26
```

Atoms specify atomic number, x/y/z and effective nuclear charge. A primitive
specifies its one-based atom index, the x/y/z powers, and its exponent.
Core primitives append their linear density coefficient. The record
`periodic` contains one flag per lattice axis. Three lattice vectors and
the image range per axis follow if an axis is periodic. Image ranges and
k-point coordinates vanish on open axes. For an isolated system use
`periodic 0 0 0`, omit the lattice and images, and supply a single Γ point.

Each `k` record gives three reciprocal fractional coordinates, weight, and
number of orbitals. Weights sum to one. Each `orbital` contains its occupation,
the number of spin components (1 or 2), and one real/imaginary coefficient
pair per primitive per component, with primitive index changing fastest.
Real pure spherical functions can be expanded into these Cartesian
primitives before export. The coefficients include the primitive and any
contraction normalization. Occupations are at most 1 for spinors and 2
for spin-independent spatial orbitals.

The Bloch basis sums translated Gaussians with
$\exp(2\pi i\,\mathbf{k}\cdot\mathbf{n})$ for integer image index **n**. The
charge density is the sum over k of the weights times the
occupation-weighted squared moduli, summed separately for each orthogonal spin
component. Positions are wrapped along periodic axes before evaluation.
The Gaussian image sum is finite. Increase `images` until the fields and the
residual face seam converge for the actual cell, exponents, and k points.
Periodic wavefunction import supplies NCI. Molecular-reference IGM and IGMH
require an explicit periodic proatom partition and are rejected. The HTML
view shows the supplied cell's atoms and the field in the sampled grid.

## Analytic sample input

A plain text file uses this versioned layout. Blank lines and text starting
with `#` are ignored. All coordinates, samples, and derivatives use the same
declared length unit. Shape, origin and the three independent Cartesian
step vectors describe an affine grid. The third grid index changes fastest.

```text
NCIPLOT-RS 1
units bohr
shape 1 1 1
origin -0.15 0.27 0.19
step 0.31 0.05 0.07
step 0.02 0.23 -0.01
step 0.08 -0.06 0.29
atoms 2
8 -0.05 -0.11 0.02 10
1 1.43 0.60 0.41 25
samples molecular with-proatoms
0.4 0.14 -0.12 0.19 0.5 -1.5 0 -1.5 0 -1
1.0 0.4 -0.3 0.6
2.0 -0.8 0.1 -0.2
```

An atom record contains atomic number, x, y, z, and a nonnegative fragment
ID. Each grid node starts with a row containing ρ, the three gradient
components, and Hessian xx, xy, xz, yy, yz, zz. With `with-proatoms`,
follow it with one row per atom containing neutral proatom ρ, gₓ, gᵧ,
and g_z. Use `density-only` for NCI without proatoms. The alternatives
`units angstrom` and `samples promolecular ...` specify Å consistently
through the derivatives, or an explicitly supplied promolecular density for
the colour and NCI coordinates. IGMH requires `molecular`. A three-dimensional
isosurface requires at least two nodes per axis.

Rust exposes `read_dataset`, `read_wfx`, `read_molden`, `Wavefunction::density_at`,
`Wavefunction::sample_grid`, `analyze`, `write_csv`, `write_cube`, and `write_html`.
`Grid::new` and `Dataset::new` also accept the data directly.

## Fields and visualisation

- NCI uses $s=|\nabla\rho|/[2(3\pi^2)^{1/3}\rho^{4/3}]$ and the middle
  eigenvalue λ₂ of the Cartesian density Hessian. The plot pairs the
  dimensionless $s$ with sign(λ₂)ρ in e/Bohr³.
- Promolecular IGM uses the free-atom gradients for Δg. For `igm` on
  molecular input, the scatter and surface colour use the supplied molecular
  density. A supplied promolecular density with its analytic Hessian supplies
  the original promolecular colour convention.
- IGMH uses stockholder weights $w_A=\rho_A^0/\sum_B\rho_B^0$.
  `igmh` follows the published maps with $w_A\nabla\rho-\rho\nabla w_A$.
  `igmh-hirshfeld` uses the product rule
  $w_A\nabla\rho+\rho\nabla w_A$. Both report the density and gradient
  partition closure defects. Their interfragment, intrafragment, and total
  Δg values are in e/Bohr⁴.
- CSV retains f64 fields, including the density gradient and Hessian
  components, with units in column names. CUBE exports a scalar volume with
  Bohr axes and third-axis-fastest order. RDG is undefined at exactly zero
  density. CSV has an empty cell there, while CUBE uses a zero placeholder.
  The separate density cube identifies those nodes.
- Open the HTML locally to rotate by drag or arrow keys and zoom by scroll
  or +/−. Use the **Clip by** menu and fractional min/max values to select an
  interval of ρ or of sign(λ₂)ρ. Signed clipping supports negative bounds.
  The colour endpoints adjust the legend and saturation independently of
  the surface and scatter selection. The scatter has a full RDG range with
  headroom and a detail option for s ≤ 2. Its horizontal density window
  is separate from surface clipping.
- The HTML embeds its driver, geometry, and scatter records. It uses
  binary-packed 32-bit mesh vertices/indices and 64-bit scatter values,
  without thinning the mesh or the points in the selected scatter window.
  Isosurfaces and density clips use linear interpolation. Browser costs
  grow with surface size and selected scatter count. Display bonds use
  covalent radii of common nonmetals within a fragment.

## Validation

```sh
cargo fmt --all --check
cargo clippy --locked --offline --all-targets -- -D warnings
RUSTFLAGS=-Dwarnings cargo test --locked --offline --all-targets -- --nocapture
```

The analytic wavefunction tests compare all Cartesian gradient and symmetric
Hessian components with independent scalar-density stencils at four steps
from 0.007 to 0.028 Bohr. They print the error, exponent, coefficient and fit
quality for every component. Observed p spans 1.99925649 to 2.00001631 with
R² at least 0.9999999807. The maximum absolute component discrepancies against
an independent 84-digit reference are 5.00 × 10⁻¹⁶ for WFX and 1.39 × 10⁻¹⁶
for Molden in atomic units. The test requires a 2 × 10⁻¹⁵ absolute threshold.

A separate molecular check used RKS dft-rs Molden exports and compared every
grid point against analytic AO-density results, with `grid_weight_cutoff=0`,
grid level 0, and one thread. The two IGMH conventions were also checked with
the same water-dimer stockholder weights and their gradients, replayed from
its exact Hirshfeld density partition to isolate the descriptor arithmetic.

| Case | Grid nodes | Max. ρ difference, e/Bohr³ | Max. Cartesian gradient difference, e/Bohr⁴ | Max. Hessian component difference, e/Bohr⁵ | Max. RDG scaled error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water dimer, ωB97X-D/STO-3G | 792 | 1.52 × 10⁻¹¹ | 1.76 × 10⁻¹⁰ | 1.70 × 10⁻⁹ | 1.30 × 10⁻¹¹ |
| Water, PBE/def2-SVP | 648 | 4.46 × 10⁻¹¹ | 2.82 × 10⁻¹⁰ | 2.32 × 10⁻⁹ | 2.36 × 10⁻¹¹ |

The largest water-dimer Δg disagreement with dft-rs was 7.11 × 10⁻¹⁵ e/Bohr⁴.
The maximum interfragment Δg on this grid was 0.01291620778887 e/Bohr⁴ for
published IGMH and 0.03978650047689 e/Bohr⁴ for the exact partition gradient.

WFN and packed-FCHK readers agreed to 5.85 × 10⁻¹¹ across the density and
nine derivative components of the same asymmetric hydrogen model, including
WFN coefficient rounding. The lithium valence and EDF fixture gave
0.1604922076790 and 0.3164460179287 e/Bohr³ at the test point. The two-k-point
expanded Gaussian chain has 1.16 occupation-weighted electrons per cell and
translational error below 5.70 × 10⁻¹⁶ across its Cartesian fields. The
independently computed density mismatch across opposing cell faces decreased
from 3.56 × 10⁻³ to 5.55 × 10⁻¹⁷ e/Bohr³ when the image range was increased
from one to four. A separate analytic evaluation reduced the largest raw
face gradient and Hessian mismatches from 1.88 × 10⁻² e/Bohr⁴ and
7.43 × 10⁻² e/Bohr⁵ to 2.22 × 10⁻¹⁶ in each unit. All 27
gradient/Hessian fits for the packed-FCHK, EDF
and non-Gamma paths at h = 0.007–0.028 Bohr had p from 1.99889309 through
2.00006817 and R² of at least 0.9999999573.

The Python wheel and source distribution were pip-installed separately
in clean Linux x86-64 CPython 3.14 environments. On the 210-node H₂
test, native Python, the pip launcher and the Rust executable produced
byte-identical CSV, CUBE and HTML products where compared. Native Windows
builds here were stopped by host application control when executing the
Cargo dependency build scripts.

The coarse-grid RDG viewer regression had 196 indexed faces and 755
points in its scatter density window. Running the embedded JavaScript with
a simulated DOM and Canvas confirmed rotation events, 57 points in the RDG
detail view, and 22 faces intersecting the signed-density window from
−0.025 to −0.004 e/Bohr³. Recolouring those faces preserved their geometry.

The water-dimer demonstration uses interfragment IGMH on a 35 × 33 × 51
grid with steps 0.18, 0.17 and 0.20 Bohr. The initial surface level is
0.005 e/Bohr⁴ and its unsigned density window is 0 to 0.10 e/Bohr³.
At the 3.3833-Bohr O···H contact, 3,533 faces lie within 1.4 Bohr of the
central 20–80% of the contact axis. The simulated initial Canvas view
painted 4,924 faces and projected the O···H separation to 59.9 pixels
at 650 × 465 pixels. dft-rs ωB97X-D/STO-3G supplied the molecular density
and stockholder weights, with 32 separately sampled
interfragment values agreeing to 8.33 × 10⁻¹⁷ e/Bohr⁴.

## References

- Johnson et al., *Revealing Noncovalent Interactions*. [JACS, 2010](https://doi.org/10.1021/ja100936w).
- Lefebvre et al., *Accurately extracting the signature of intermolecular interactions present in the NCI plot of the reduced density gradient versus electron density*. [PCCP, 2017](https://doi.org/10.1039/C7CP02110K).
- Lu and Chen, *Independent gradient model based on Hirshfeld partition*. [Journal of Computational Chemistry, 2022](https://doi.org/10.1002/jcc.26812).
- Lu and Chen, *Erratum to Independent gradient model based on Hirshfeld partition*. [ChemRxiv](https://doi.org/10.26434/chemrxiv-2022-g1m34).
- Cordero et al., *Covalent radii revisited*. [Dalton Transactions, 2008](https://doi.org/10.1039/B801115J).

Apache-2.0. [License](LICENSE) and [attribution](NOTICE).
