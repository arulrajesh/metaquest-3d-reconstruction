"""Automatic ``adb reverse`` so a USB-connected headset can reach the laptop at 127.0.0.1."""

from __future__ import annotations

import logging
import re
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


ENDPOINT_FILE_NAME = "live_stream_endpoint.json"
_IP_PORT = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}):\d+$")
_INET = re.compile(r"inet (\d{1,3}(?:\.\d{1,3}){3})/")


@dataclass(frozen=True)
class AdbDeviceSetup:
    serial: str
    reverse_ok: bool
    device_ip: str | None
    endpoint_host: str | None
    endpoint_pushed: bool


def device_ip(adb: Path, serial: str) -> str | None:
    """Headset Wi-Fi address: from an ``ip:port`` wireless serial, else from ``ip addr`` on wlan0."""
    match = _IP_PORT.match(serial)
    if match:
        return match.group(1)
    try:
        result = _run(adb, "-s", serial, "shell", "ip", "-f", "inet", "addr", "show", "wlan0")
    except (OSError, subprocess.SubprocessError):
        return None
    found = _INET.search(result.stdout)
    return found.group(1) if found else None


def local_ip_towards(remote_ip: str) -> str | None:
    """The laptop address the OS would use to reach ``remote_ip`` (no packets are sent)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((remote_ip, 9))
            return str(sock.getsockname()[0])
    except OSError:
        return None


def push_endpoint(adb: Path, serial: str, host: str, port: int, package: str) -> bool:
    import json
    import tempfile

    payload = json.dumps({"host": host, "port": port})
    remote = f"/sdcard/Android/data/{package}/files/{ENDPOINT_FILE_NAME}"
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / ENDPOINT_FILE_NAME
        local.write_text(payload, encoding="utf-8")
        try:
            result = _run(adb, "-s", serial, "push", str(local), remote)
        except (OSError, subprocess.SubprocessError):
            return False
    return result.returncode == 0


def setup_adb_connection(
    tcp_port: int,
    *,
    adb_path: Path | None = None,
    push: bool = True,
    package: str = "com.arulrajesh.QuestRealityCaptureLive",
) -> tuple[Path | None, list[AdbDeviceSetup]]:
    """For every adb device: ``adb reverse`` plus (wireless) push of the laptop endpoint. Never raises."""
    adb = find_adb(adb_path)
    if adb is None:
        return None, []
    try:
        serials = list_devices(adb)
    except (OSError, subprocess.SubprocessError):
        return adb, []
    setups = []
    for serial in serials:
        try:
            reverse = _run(adb, "-s", serial, "reverse", f"tcp:{tcp_port}", f"tcp:{tcp_port}").returncode == 0
        except (OSError, subprocess.SubprocessError):
            reverse = False
        ip = device_ip(adb, serial)
        host = local_ip_towards(ip) if ip else None
        pushed = bool(push and host and push_endpoint(adb, serial, host, tcp_port, package))
        setups.append(AdbDeviceSetup(serial=serial, reverse_ok=reverse, device_ip=ip, endpoint_host=host, endpoint_pushed=pushed))
    return adb, setups


class AdbReverseKeeper:
    """Keep ``adb reverse`` forwards (and wireless adb) alive while a local server runs.

    A sleeping Quest drops off wireless adb, its ``adb reverse`` forwards go with it, and it does not
    reconnect on its own when it wakes: ``http://localhost:8080`` in the Quest browser then stops working.
    Every ``interval_s`` this reconnects wireless headsets seen earlier (``adb connect``, at most every
    ``reconnect_interval_s``) and re-applies missing forwards. Runs in a daemon thread and never raises.
    ``runner(args) -> (returncode, stdout)`` is injectable for tests.
    """

    def __init__(
        self,
        ports: list[int],
        *,
        adb_path: Path | None = None,
        interval_s: float = 5.0,
        reconnect_interval_s: float = 15.0,
        on_change=None,
        runner=None,
        clock=None,
    ):
        import time

        self.ports = list(ports)
        self.interval_s = interval_s
        self.reconnect_interval_s = reconnect_interval_s
        self.on_change = on_change or (lambda text: logger.info(text))
        self._adb = find_adb(adb_path)
        self._runner = runner or self._subprocess_runner
        self._clock = clock or time.monotonic
        self._wireless: set[str] = set()
        self._last_connect: dict[str, float] = {}
        self._stop = None
        self._thread = None

    def _subprocess_runner(self, args: list[str]) -> tuple[int, str]:
        if self._adb is None:
            return 1, ""
        try:
            result = subprocess.run([str(self._adb), *args], capture_output=True, text=True, timeout=8)
            return result.returncode, result.stdout
        except (OSError, subprocess.SubprocessError):
            return 1, ""

    def tick(self) -> list[str]:
        """One pass; returns what changed (also passed to ``on_change``)."""
        changes: list[str] = []
        code, out = self._runner(["devices"])
        if code != 0:
            return changes
        states = {}
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
        ready = [serial for serial, state in states.items() if state == "device"]
        self._wireless.update(serial for serial in ready if re.fullmatch(r"[\d.]+:\d+", serial))
        now = self._clock()
        for serial in sorted(self._wireless - set(ready)):
            if now - self._last_connect.get(serial, -1e9) < self.reconnect_interval_s:
                continue
            self._last_connect[serial] = now
            if states.get(serial) == "offline":  # stale entry: "adb connect" alone says "already connected"
                self._runner(["disconnect", serial])
            code, out = self._runner(["connect", serial])
            if code == 0 and "connected to" in out:
                changes.append(f"headset {serial} reconnected")
                ready.append(serial)
        for serial in ready:
            code, out = self._runner(["-s", serial, "reverse", "--list"])
            if code != 0:
                continue
            for port in self.ports:
                if f"tcp:{port} tcp:{port}" in out:
                    continue
                code, _ = self._runner(["-s", serial, "reverse", f"tcp:{port}", f"tcp:{port}"])
                if code == 0:
                    changes.append(f"headset {serial}: adb reverse tcp:{port} restored")
        for change in changes:
            self.on_change(change)
        return changes

    def start(self) -> "AdbReverseKeeper":
        import threading

        if self._thread is not None:
            return self
        self._stop = threading.Event()

        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                try:
                    self.tick()
                except Exception as error:  # never take the server down
                    logger.debug("adb keeper pass failed: %s", error)

        self._thread = threading.Thread(target=loop, name="mq3drecon-adb-keeper", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
