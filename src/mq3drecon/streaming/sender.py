"""TCP sender for the live streaming protocol (used by replay and tests)."""

from __future__ import annotations

import socket
from types import TracebackType

from mq3drecon.streaming.messages import Message, encode_message

_SEND_BUFFER_BYTES = 8 * 1024 * 1024


class LiveSender:
    def __init__(self, host: str, port: int, *, timeout_s: float = 5.0):
        self.host = host
        self.port = port
        self._timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self.bytes_sent = 0

    def connect(self) -> "LiveSender":
        sock = socket.create_connection((self.host, self.port), timeout=self._timeout_s)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SEND_BUFFER_BYTES)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        self._socket = sock
        return self

    def send(self, message: Message) -> None:
        if self._socket is None:
            raise ConnectionError("LiveSender is not connected")
        data = encode_message(message)
        self._socket.sendall(data)
        self.bytes_sent += len(data)

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "LiveSender":
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
