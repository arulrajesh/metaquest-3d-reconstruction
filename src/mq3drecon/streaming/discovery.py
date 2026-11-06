"""Zero-configuration discovery of the laptop receiver on the local network.

The headset broadcasts a UDP probe (``MQLS?``) to the discovery port. The
laptop's :class:`DiscoveryResponder` answers with a unicast JSON reply naming
its TCP port. Headset-initiated discovery is used because unicast replies are
delivered reliably to Android devices, while inbound broadcasts may be filtered.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass

from mq3drecon.streaming.framing import PROTOCOL_VERSION

logger = logging.getLogger(__name__)

DISCOVERY_PORT = 9871
PROBE_PREFIX = b"MQLS?"
SERVICE_NAME = "mq3drecon-live"


@dataclass(frozen=True)
class DiscoveredReceiver:
    address: str
    port: int
    host_name: str


def build_reply(tcp_port: int, host_name: str) -> bytes:
    return json.dumps(
        {"service": SERVICE_NAME, "protocol_version": PROTOCOL_VERSION, "port": tcp_port, "host": host_name}
    ).encode("utf-8")


def parse_reply(data: bytes, address: str) -> DiscoveredReceiver | None:
    try:
        reply = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(reply, dict) or reply.get("service") != SERVICE_NAME:
        return None
    if int(reply.get("protocol_version", -1)) != PROTOCOL_VERSION:
        return None
    return DiscoveredReceiver(address=address, port=int(reply["port"]), host_name=str(reply.get("host", "")))


class DiscoveryResponder:
    """Answers discovery probes until stopped."""

    def __init__(self, tcp_port: int, *, discovery_port: int = DISCOVERY_PORT, host_name: str | None = None):
        self._reply = build_reply(tcp_port, host_name or socket.gethostname())
        self._discovery_port = discovery_port
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.port = discovery_port

    def start(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self._discovery_port))
        sock.settimeout(0.25)
        self._socket = sock
        self.port = int(sock.getsockname()[1])
        self._thread = threading.Thread(target=self._loop, name="mq3drecon-live-discovery", daemon=True)
        self._thread.start()
        logger.info("Discovery responder on UDP %d", self.port)
        return self.port

    def stop(self) -> None:
        self._stopping.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        assert self._socket is not None
        while not self._stopping.is_set():
            try:
                data, address = self._socket.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.startswith(PROBE_PREFIX):
                try:
                    self._socket.sendto(self._reply, address)
                except OSError as exc:
                    logger.debug("Discovery reply to %s failed: %s", address, exc)


def discover_receivers(
    *,
    timeout_s: float = 1.5,
    discovery_port: int = DISCOVERY_PORT,
    target: str = "255.255.255.255",
) -> list[DiscoveredReceiver]:
    """Send a probe and collect replies (the headset does the same thing in C#)."""
    found: dict[tuple[str, int], DiscoveredReceiver] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)
        sock.sendto(PROBE_PREFIX, (target, discovery_port))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data, (address, _port) = sock.recvfrom(1024)
            except socket.timeout:
                continue
            receiver = parse_reply(data, address)
            if receiver is not None:
                found[(receiver.address, receiver.port)] = receiver
    return list(found.values())
