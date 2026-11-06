"""TCP sender for the live streaming protocol (used by replay and tests)."""

from __future__ import annotations

import queue
import socket
import threading
from types import TracebackType

from mq3drecon.streaming.messages import Message, encode_message

_SEND_BUFFER_BYTES = 8 * 1024 * 1024


class LiveSender:
    def __init__(self, host: str, port: int, *, timeout_s: float = 5.0):
        self.host = host
        self.port = port
        self._timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._inbox: queue.Queue = queue.Queue()
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

    def receive(self, timeout_s: float | None = None):
        """Next message sent back by the receiver (feedback), or None on timeout or after the link closed.

        A background thread does blocking reads (socket timeouts would corrupt a buffered reader).
        """
        if self._socket is None:
            raise ConnectionError("LiveSender is not connected")
        if self._reader is None:
            self._reader = threading.Thread(target=self._read_loop, name="mq3drecon-live-sender-reader", daemon=True)
            self._reader.start()
        try:
            return self._inbox.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def _read_loop(self) -> None:
        from mq3drecon.streaming.framing import StreamProtocolError, read_frame
        from mq3drecon.streaming.messages import decode_message

        assert self._socket is not None
        stream = self._socket.makefile("rb")
        try:
            while True:
                raw = read_frame(stream)
                if raw is None:
                    return
                self._inbox.put(decode_message(raw))
        except (OSError, StreamProtocolError):
            return

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
