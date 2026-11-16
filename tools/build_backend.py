# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Portable wheel building with maturin.

The build-root mapping and SBOM identity normalization follow the dft-rs
packaging convention. Dependency metadata stays in the wheel.
"""

import base64
import csv
import hashlib
import io
import json
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import maturin

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _compiler_paths():
    home = Path.home()
    roots = {
        Path(os.environ.get("CARGO_HOME", home / ".cargo")).resolve(): "cargo",
        Path(os.environ.get("RUSTUP_HOME", home / ".rustup")).resolve(): "rustup",
        home.resolve(): "user",
        ROOT: "nciplot-rs",
    }
    if target := os.environ.get("CARGO_TARGET_DIR"):
        roots[Path(target).resolve()] = "build"
    key = "CARGO_ENCODED_RUSTFLAGS"
    before = os.environ.get(key)
    flags = before.split("\x1f") if before else os.environ.get("RUSTFLAGS", "").split()
    substitutions = [f"--remap-path-prefix={path}={public}" for path, public in roots.items()]
    os.environ[key] = "\x1f".join([*flags, *substitutions])
    try:
        yield roots
    finally:
        if before is None:
            del os.environ[key]
        else:
            os.environ[key] = before


def _root_spellings(roots):
    spellings = {}
    for path, public in roots.items():
        for private, replacement in (
            (str(path), public), (path.as_posix(), public),
            (path.as_uri(), "file:./" + public),
        ):
            for safe in ("", "/:"):
                spellings[quote(private, safe=safe)] = quote(replacement, safe=safe)
            spellings[private] = replacement
    return sorted(spellings.items(), key=lambda item: -len(item[0]))


def _clean_json(data, spellings):
    def visit(value):
        if isinstance(value, str):
            for private, public in spellings:
                value = value.replace(private, public)
        elif isinstance(value, dict):
            value = {key: visit(entry) for key, entry in value.items()}
        elif isinstance(value, list):
            value = [visit(entry) for entry in value]
        return value

    return json.dumps(visit(json.loads(data)), ensure_ascii=False, separators=(",", ":")).encode()


def _record_row(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return name, "sha256=" + digest, len(data)


def _finalize(path, roots):
    spellings = _root_spellings(roots)
    markers = [value.encode(encoding) for value, _ in spellings
               for encoding in ("utf-8", "utf-16-le")]
    staged = path.with_suffix(".staged")
    try:
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(staged, "w") as out:
            names = source.namelist()
            if len(set(names)) != len(names):
                raise ValueError("wheel contains duplicate members")
            records = [name for name in names if name.endswith(".dist-info/RECORD")]
            if len(records) != 1:
                raise ValueError("wheel requires a single RECORD")
            record = records[0]
            index = io.StringIO(newline="")
            writer = csv.writer(index, lineterminator="\n")
            for info in source.infolist():
                name = info.filename
                parsed = PurePosixPath(name)
                if parsed.is_absolute() or ".." in parsed.parts:
                    raise ValueError("wheel path escapes the archive")
                if name == record:
                    continue
                data = source.read(info)
                if ".dist-info/sboms/" in name and name.endswith(".json"):
                    data = _clean_json(data, spellings)
                if any(marker in data for marker in markers):
                    raise ValueError(f"private build path found in wheel member {name}")
                out.writestr(info, data)
                writer.writerow(_record_row(name, data))
            writer.writerow((record, "", ""))
            out.writestr(source.getinfo(record), index.getvalue().encode())
        staged.replace(path)
    finally:
        if staged.exists():
            staged.unlink()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Compile and normalize the wheel before it reaches the installer."""
    with _compiler_paths() as roots:
        filename = maturin.build_wheel(wheel_directory, config_settings, metadata_directory)
        _finalize(Path(wheel_directory) / filename, roots)
    return filename


def build_sdist(sdist_directory, config_settings=None):
    """Build the whitelisted source archive."""
    return maturin.build_sdist(sdist_directory, config_settings)
