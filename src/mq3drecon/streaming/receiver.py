"""TCP receiver for the live streaming protocol (laptop side)."""

from __future__ import annotations

import logging
import socket
import sys
import threading
from typing import Protocol

from mq3drecon.streaming.framing import StreamProtocolError, read_frame
from mq3drecon.streaming.messages import Message, decode_message

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9870
_RECEIVE_BUFFER_BYTES = 8 * 1024 * 1024


class LiveStreamHandler(Protocol):
    """Receives decoded messages. Calls are serialized: never two at once."""

    def on_connect(self, peer: str) -> None: ...

    def on_message(self, message: Message) -> None: ...

    def on_disconnect(self, peer: str, error: BaseException | None) -> None: ...


class LiveReceiver:
    """Accepts one headset connection at a time.

    A newer connection replaces an older one (the headset reconnecting after a
    network drop must not be locked out by a half-open stale socket). The old
    connection is fully closed and its ``on_disconnect`` delivered before the
    new connection's ``on_connect``.
    """

    def __init__(
        self,
        handler: LiveStreamHandler,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        *,
        idle_timeout_s: float = 10.0,
    ):
        self._handler = handler
        self._host = host
        self._requested_port = port
        self._idle_timeout_s = idle_timeout_s
        self._server: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._connection: socket.socket | None = None
        self._handler_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stopping = threading.Event()
        self.port = port

    def start(self) -> int:
        """Bind and start accepting. Returns the bound port (useful with ``port=0``)."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform != "win32":
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._requested_port))
        server.listen(2)
        server.settimeout(0.25)
        self._server = server
        self.port = int(server.getsockname()[1])
        self._accept_thread = threading.Thread(target=self._accept_loop, name="mq3drecon-live-accept", daemon=True)
        self._accept_thread.start()
        logger.info("Live receiver listening on %s:%d", self._host, self.port)
        return self.port

    def stop(self) -> None:
        self._stopping.set()
        if self._server is not None:
            self._server.close()
        self._close_current_connection()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stopping.is_set():
            try:
                connection, address = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._close_current_connection()
            self._start_reader(connection, f"{address[0]}:{address[1]}")

    def _start_reader(self, connection: socket.socket, peer: str) -> None:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECEIVE_BUFFER_BYTES)
        connection.settimeout(self._idle_timeout_s)
        with self._state_lock:
            self._connection = connection
            self._reader_thread = threading.Thread(
                target=self._read_loop, args=(connection, peer), name="mq3drecon-live-reader", daemon=True
            )
            self._reader_thread.start()

    def _close_current_connection(self) -> None:
        with self._state_lock:
            connection, thread = self._connection, self._reader_thread
            self._connection, self._reader_thread = None, None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _read_loop(self, connection: socket.socket, peer: str) -> None:
        error: BaseException | None = None
        with self._handler_lock:
            self._handler.on_connect(peer)
        try:
            stream = connection.makefile("rb", buffering=1024 * 1024)
            while not self._stopping.is_set():
                raw = read_frame(stream)
                if raw is None:
                    break
                message = decode_message(raw)
                with self._handler_lock:
                    self._handler.on_message(message)
        except socket.timeout:
            error = TimeoutError(f"No data from {peer} for {self._idle_timeout_s:.0f} s")
            logger.warning("%s", error)
        except StreamProtocolError as exc:
            error = exc
            logger.error("Protocol error from %s: %s", peer, exc)
        except OSError as exc:
            # Closed locally (replacement or stop) or reset by the peer.
            error = None if self._stopping.is_set() else exc
        finally:
            try:
                connection.close()
            except OSError:
                pass
            with self._handler_lock:
                self._handler.on_disconnect(peer, error)
