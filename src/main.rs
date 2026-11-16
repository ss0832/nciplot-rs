// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

fn main() -> std::process::ExitCode {
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let code = nciplot_rs::run_cli(std::env::args_os().skip(1), stdout.lock(), stderr.lock());
    match code {
        Ok(value) => value.into(),
        Err(error) => {
            eprintln!("nciplot-rs: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
