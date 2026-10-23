# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Windows CUDA runtime discovery without loading a DLL."""

import os
import struct
import sysconfig
from pathlib import Path
from shutil import which


def find_cuda_runtime(cuda_version: str | None) -> str:
    """Find cudart for the Python process, including x64 emulation on ARM64."""
    override = os.environ.get("VLLM_CUDART_SO_PATH")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"VLLM_CUDART_SO_PATH is not a file: {path}")
        return str(path.resolve())

    platform = sysconfig.get_platform().lower()
    architectures = {"win-arm64": "arm64", "win-amd64": "x64"}
    if platform not in architectures:
        raise ValueError(f"Unsupported Windows Python platform: {platform}")
    architecture = architectures[platform]
    major = int((cuda_version or "12").split(".")[0])
    suffix = str(major) if major >= 12 else f"{major}0"
    filename = f"cudart64_{suffix}.dll"

    cuda_root = next(
        (
            os.environ[name]
            for name in ("CUDA_HOME", "CUDA_ROOT", "CUDA_PATH")
            if os.environ.get(name)
        ),
        None,
    )
    candidates = []
    if cuda_root:
        bindir = Path(cuda_root) / "bin"
        candidates.extend((bindir / architecture / filename, bindir / filename))
    else:
        compiler = which("nvcc")
        if compiler:
            bindir = Path(compiler).resolve().parent
            # A compiler on PATH can target the other process architecture.
            if bindir.name.lower() in ("arm64", "x64"):
                bindir = bindir.parent
            candidates.extend((bindir / architecture / filename, bindir / filename))
    rejected = []
    expected_machine = {"arm64": 0xAA64, "x64": 0x8664}[architecture]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("rb") as stream:
                dos = stream.read(64)
                if len(dos) != 64 or dos[:2] != b"MZ":
                    raise ValueError("missing DOS header")
                stream.seek(struct.unpack_from("<I", dos, 60)[0])
                pe = stream.read(6)
                if len(pe) != 6 or pe[:4] != b"PE" + bytes(2):
                    raise ValueError("missing PE header")
                machine = struct.unpack_from("<H", pe, 4)[0]
            if machine != expected_machine:
                raise ValueError(
                    f"PE machine 0x{machine:04x}, expected 0x{expected_machine:04x}"
                )
        except (OSError, ValueError) as error:
            rejected.append(f"{candidate}: {error}")
            continue
        return str(candidate.resolve())
    searched = ", ".join(map(str, candidates)) or "no CUDA root or nvcc on PATH"
    raise FileNotFoundError(
        f"Cannot find {filename} for {platform}; searched: {searched}. "
        "Set VLLM_CUDART_SO_PATH to the compatible CUDA runtime DLL. "
        + "; ".join(rejected)
    )
