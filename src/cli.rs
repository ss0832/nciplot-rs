// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use crate::{
    analyze, read_dataset, write_csv, write_cube, write_html, CubeField, Dataset, Descriptor,
    Error, Method, Result, SurfaceDensity, ViewerOptions, WavefunctionFormat,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

const HELP: &str = r"nciplot-rs
Analyze real-space molecular interaction fields and render a single offline HTML viewer.

USAGE
    nciplot-rs METHOD INPUT [OPTIONS]

METHOD
    nci | igm | igmh | igmh-hirshfeld

OUTPUT
    --prefix PATH           Prefix for CSV, CUBE and HTML files (default: interaction)
    --field rdg|total|inter|intra
                            Surface/descriptor CUBE field (default: rdg for NCI,
                            inter for multiple fragments, otherwise total)
    --isovalue VALUE        Surface level (0.5 for RDG or 0.005 e/Bohr^4)
    --surface-density rho|signed
                            Density quantity for surface clipping (default: rho)
    --density-min VALUE     Minimum selected density on surface, e/Bohr^3
    --density-max VALUE     Maximum selected density on surface, e/Bohr^3
    --color-limit VALUE     Symmetric signed-density color range, e/Bohr^3
    --color-min VALUE       Lower surface color endpoint, e/Bohr^3
    --color-max VALUE       Upper surface color endpoint, e/Bohr^3
    --scatter-limit VALUE   Symmetric signed-density scatter x range, e/Bohr^3

WAVEFUNCTION INPUT
    Extensions .wfx, .molden, .molden.input, .wfn, .fchk, .fch or .ncwf select
    wavefunction import. Other paths use the analytic-sample format.
    --shape NX,NY,NZ        Number of grid nodes per axis (default: 41,41,41)
    --padding VALUE         Grid margin beyond the nuclei (default: 5 Bohr)
    --fragments A,B,...     Fragment ID in input atom order (default: all zero)
    --proatom Z=FILE        Neutral atomic wavefunction reference, repeat per element
";

struct Config {
    method: Method,
    input: PathBuf,
    prefix: PathBuf,
    field: Option<Descriptor>,
    isovalue: Option<f64>,
    surface_density: SurfaceDensity,
    density_min: Option<f64>,
    density_max: Option<f64>,
    color_limit: Option<f64>,
    color_min: Option<f64>,
    color_max: Option<f64>,
    scatter_limit: Option<f64>,
    shape: Option<[usize; 3]>,
    padding: Option<f64>,
    fragments: Option<Vec<usize>>,
    proatoms: BTreeMap<u8, PathBuf>,
}

fn error(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

fn text_arg(value: &OsString) -> Result<&str> {
    value
        .to_str()
        .ok_or_else(|| error("flag or value is not valid Unicode"))
}

fn value<T: std::str::FromStr>(arg: &OsString) -> Result<T> {
    text_arg(arg)?
        .parse()
        .map_err(|_| error(format!("invalid option value {}", arg.to_string_lossy())))
}

fn integer_list(arg: &OsString) -> Result<Vec<usize>> {
    text_arg(arg)?
        .split(',')
        .map(|part| {
            part.parse()
                .map_err(|_| error(format!("invalid integer in list: {part}")))
        })
        .collect()
}

impl Config {
    fn parse(mut args: impl Iterator<Item = OsString>, method: Method) -> Result<Self> {
        let input = args.next().ok_or_else(|| error("expected input path"))?;
        let mut config = Self {
            method,
            input: input.into(),
            prefix: PathBuf::from("interaction"),
            field: None,
            isovalue: None,
            surface_density: SurfaceDensity::Unsigned,
            density_min: None,
            density_max: None,
            color_limit: None,
            color_min: None,
            color_max: None,
            scatter_limit: None,
            shape: None,
            padding: None,
            fragments: None,
            proatoms: BTreeMap::new(),
        };
        while let Some(flag) = args.next() {
            let key = text_arg(&flag)?;
            let arg = args
                .next()
                .ok_or_else(|| error(format!("missing value for {key}")))?;
            match key {
                "--prefix" => config.prefix = arg.into(),
                "--field" => config.field = Some(text_arg(&arg)?.parse()?),
                "--isovalue" => config.isovalue = Some(value(&arg)?),
                "--surface-density" => {
                    config.surface_density = match text_arg(&arg)? {
                        "rho" => SurfaceDensity::Unsigned,
                        "signed" => SurfaceDensity::Signed,
                        _ => return Err(error("surface density must be rho or signed")),
                    };
                }
                "--density-min" => config.density_min = Some(value(&arg)?),
                "--density-max" => config.density_max = Some(value(&arg)?),
                "--color-limit" => config.color_limit = Some(value(&arg)?),
                "--color-min" => config.color_min = Some(value(&arg)?),
                "--color-max" => config.color_max = Some(value(&arg)?),
                "--scatter-limit" => config.scatter_limit = Some(value(&arg)?),
                "--shape" => {
                    config.shape = Some(
                        integer_list(&arg)?
                            .try_into()
                            .map_err(|_| error("shape requires exactly three node counts"))?,
                    )
                }
                "--padding" => config.padding = Some(value(&arg)?),
                "--fragments" => config.fragments = Some(integer_list(&arg)?),
                "--proatom" => {
                    let (element, path) = text_arg(&arg)?
                        .split_once('=')
                        .ok_or_else(|| error("proatom must have the form Z=FILE"))?;
                    let element = element
                        .parse::<u8>()
                        .ok()
                        .filter(|z| (1..=118).contains(z))
                        .ok_or_else(|| {
                            error("proatom element must be an atomic number 1 through 118")
                        })?;
                    if config.proatoms.insert(element, path.into()).is_some() {
                        return Err(error(format!(
                            "multiple proatom references for atomic number {element}"
                        )));
                    }
                }
                _ => return Err(error(format!("unknown option {key}"))),
            }
        }
        Ok(config)
    }

    fn dataset(&self) -> Result<Dataset> {
        let input = BufReader::new(File::open(&self.input)?);
        if let Some(format) = WavefunctionFormat::detect(&self.input) {
            let wavefunction = format.read(input)?;
            let grid = wavefunction
                .bounding_grid(self.shape.unwrap_or([41; 3]), self.padding.unwrap_or(5.0))?;
            let references = self
                .proatoms
                .iter()
                .map(|(&element, path)| {
                    let format = WavefunctionFormat::detect(path).ok_or_else(|| {
                        error("proatom reference requires a supported wavefunction extension")
                    })?;
                    let reference = format.read(BufReader::new(File::open(path)?))?;
                    Ok((element, reference))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            wavefunction.sample_grid(
                grid,
                self.fragments.as_deref(),
                (!references.is_empty()).then_some(&references),
            )
        } else {
            if self.shape.is_some()
                || self.padding.is_some()
                || self.fragments.is_some()
                || !self.proatoms.is_empty()
            {
                return Err(error("grid shape, padding, fragments and reference options require wavefunction input"));
            }
            read_dataset(input)
        }
    }

    fn viewer_options(&self, dataset: &Dataset) -> ViewerOptions {
        let descriptor = self
            .field
            .unwrap_or_else(|| self.method.default_descriptor(dataset));
        let mut options = ViewerOptions::for_density(descriptor, self.surface_density);
        options.isovalue = self.isovalue.unwrap_or(options.isovalue);
        options.surface_density_min = self.density_min.unwrap_or(options.surface_density_min);
        options.surface_density_max = self.density_max.unwrap_or(options.surface_density_max);
        if let Some(limit) = self.color_limit {
            options.color_min = -limit;
            options.color_max = limit;
        }
        options.color_min = self.color_min.unwrap_or(options.color_min);
        options.color_max = self.color_max.unwrap_or(options.color_max);
        options.scatter_density_limit = self.scatter_limit.unwrap_or(options.scatter_density_limit);
        options
    }
}

fn product_path(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = prefix.as_os_str().to_owned();
    path.push(suffix);
    path.into()
}

fn output(
    prefix: &Path,
    suffix: &str,
    stdout: &mut impl Write,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    let path = product_path(prefix, suffix);
    let mut out = BufWriter::new(File::create(&path)?);
    write(&mut out)?;
    out.flush()?;
    writeln!(stdout, "{}", path.display())?;
    Ok(())
}

fn run(
    mut args: impl Iterator<Item = OsString>,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> Result<()> {
    let command = args.next();
    match command.as_ref().map(text_arg).transpose()? {
        None | Some("--help" | "-h") => write!(stdout, "{HELP}")?,
        Some("--version" | "-V") => writeln!(stdout, "nciplot-rs {}", env!("CARGO_PKG_VERSION"))?,
        Some(method) => {
            let config = Config::parse(args, method.parse()?)?;
            let dataset = config.dataset()?;
            let analysis = analyze(&dataset, config.method)?;
            let options = config.viewer_options(&dataset);
            output(&config.prefix, ".html", stdout, |out| {
                write_html(out, &dataset, &analysis, options)
            })?;
            output(&config.prefix, ".csv", stdout, |out| {
                write_csv(out, &dataset, &analysis)
            })?;
            output(&config.prefix, ".field.cube", stdout, |out| {
                write_cube(
                    out,
                    &dataset,
                    &analysis,
                    CubeField::Descriptor(options.descriptor),
                )
            })?;
            output(&config.prefix, ".signed-density.cube", stdout, |out| {
                write_cube(out, &dataset, &analysis, CubeField::SignedDensity)
            })?;
            output(&config.prefix, ".density.cube", stdout, |out| {
                write_cube(out, &dataset, &analysis, CubeField::Density)
            })?;
            writeln!(
                stderr,
                "{} nodes, {} zero-density nodes, density model {}",
                analysis.points.len(),
                analysis.zero_density_points,
                dataset.density_model()
            )?;
            if let (Some(density), Some(gradient)) =
                (analysis.density_residual, analysis.gradient_residual)
            {
                writeln!(stderr, "partition closure: density {density:.6e} e/Bohr^3, gradient {gradient:.6e} e/Bohr^4")?;
            }
        }
    }
    Ok(())
}

/// Execute the command-line interface with caller-owned output streams.
pub fn run_cli(
    args: impl IntoIterator<Item = OsString>,
    mut stdout: impl Write,
    mut stderr: impl Write,
) -> std::io::Result<u8> {
    match run(args.into_iter(), &mut stdout, &mut stderr) {
        Ok(()) => Ok(0),
        Err(error) => {
            writeln!(stderr, "nciplot-rs: {error}")?;
            Ok(1)
        }
    }
}
