"""Automatic ``adb reverse`` so a USB-connected headset can reach the laptop at 127.0.0.1."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdbReverseResult:
    adb_path: Path | None
    devices: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.devices)


def find_adb(explicit: Path | None = None) -> Path | None:
    """Locate adb: explicit path, then Android SDK installs, then PATH.

    SDK installs are preferred over PATH because tools bundle old adb builds
    (on this laptop Touch Portal ships adb 28, which predates Quest wireless
    debugging), and two adb versions kill each other's server.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidates = []
    for env_name in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]) / "platform-tools")
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Android" / "Sdk" / "platform-tools")
    for directory in candidates:
        for name in ("adb.exe", "adb"):
            if (directory / name).exists():
                return directory / name
    on_path = shutil.which("adb")
    return Path(on_path) if on_path else None


def _run(adb: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(adb), *args], capture_output=True, text=True, timeout=15)


def list_devices(adb: Path) -> list[str]:
    result = _run(adb, "devices")
    serials = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])
    return serials


def setup_adb_reverse(tcp_port: int, *, adb_path: Path | None = None) -> AdbReverseResult:
    """Run ``adb reverse tcp:PORT tcp:PORT`` for every connected device. Never raises."""
    adb = find_adb(adb_path)
    if adb is None:
        return AdbReverseResult(adb_path=None, detail="adb not found; USB streaming disabled (Wi-Fi still works)")
    try:
        serials = list_devices(adb)
        configured = []
        for serial in serials:
            result = _run(adb, "-s", serial, "reverse", f"tcp:{tcp_port}", f"tcp:{tcp_port}")
            if result.returncode == 0:
                configured.append(serial)
            else:
                logger.warning("adb reverse failed for %s: %s", serial, result.stderr.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        return AdbReverseResult(adb_path=adb, detail=f"adb failed: {exc}")
    if not serials:
        return AdbReverseResult(adb_path=adb, detail="no USB headset detected (Wi-Fi discovery still works)")
    return AdbReverseResult(adb_path=adb, devices=configured, detail=f"adb reverse tcp:{tcp_port} on {configured}")
