# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Python interface to real-space NCI, IGM and IGMH analysis."""

import sys

from ._native import (
    Analysis,
    Dataset,
    Grid,
    ViewerOptions,
    Wavefunction,
    __version__,
    _cli,
    load_dataset,
    load_wavefunction,
)

__all__ = [
    "Analysis",
    "Dataset",
    "Grid",
    "ViewerOptions",
    "Wavefunction",
    "__version__",
    "load_dataset",
    "load_wavefunction",
]


def cli_main():
    """Run the same driver as the Rust nciplot-rs command."""
    status, stdout, stderr = _cli(sys.argv[1:])
    sys.stdout.buffer.write(stdout)
    sys.stderr.buffer.write(stderr)
    return status
