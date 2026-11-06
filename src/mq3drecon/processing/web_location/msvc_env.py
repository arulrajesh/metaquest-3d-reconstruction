"""Make the MSVC compiler available to PyTorch CUDA extension builds on Windows (gsplat JIT-compiles once).

Finds Visual Studio Build Tools with the C++ toolset via ``vswhere``, runs
``vcvars64.bat``, and copies the resulting environment into ``os.environ``.
No-op on other platforms or when ``cl.exe`` is already on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"


def ensure_msvc_env() -> bool:
    """Return True when ``cl.exe`` is usable afterwards."""
    if sys.platform != "win32" or shutil.which("cl"):
        return True
    if not _VSWHERE.exists():
        return False
    found = subprocess.run(
        [str(_VSWHERE), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    vcvars = Path(found) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if found else None
    if vcvars is None or not vcvars.exists():
        return False
    output = subprocess.run(f'cmd /s /c ""{vcvars}" >nul && set"', capture_output=True, text=True, shell=True).stdout
    for line in output.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key] = value
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")
    return shutil.which("cl") is not None


def _msvc_compatible(flags: list[str] | None) -> list[str]:
    """Drop or translate GCC/Clang-only host compiler flags (MSVC reads ``-Wxyz`` as ``/W<level>`` and fails)."""
    cleaned: list[str] = []
    skip_next = False
    for flag in flags or []:
        if skip_next:
            skip_next = False
            continue
        if flag == "-include":  # precompiled-header include + its path
            skip_next = True
            continue
        if flag.startswith("-W"):
            continue
        cleaned.append({"-O3": "/O2", "-O2": "/O2", "-O0": "/Od"}.get(flag, flag))
    return cleaned


def prepare_gsplat_build() -> bool:
    """Call before importing gsplat: MSVC environment + MSVC-safe host flags for its one-time CUDA JIT build."""
    ok = ensure_msvc_env()
    if sys.platform != "win32":
        return ok
    import torch.utils.cpp_extension as cpp_extension

    original = cpp_extension._jit_compile
    if getattr(original, "_mq3drecon_msvc", False):
        return ok

    def jit_compile(name, sources, extra_cflags, *args, **kwargs):
        return original(name, sources, _msvc_compatible(extra_cflags), *args, **kwargs)

    jit_compile._mq3drecon_msvc = True  # type: ignore[attr-defined]
    cpp_extension._jit_compile = jit_compile
    return ok
