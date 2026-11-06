"""TCP receiver for the live streaming protocol (laptop side)."""

from __future__ import annotations

import logging
import socket
import sys
import threading
from typing import Protocol

from mq3drecon.streaming.framing import StreamProtocolError, read_frame
from mq3drecon.streaming.messages import Message, decode_message, encode_message

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9870
_RECEIVE_BUFFER_BYTES = 8 * 1024 * 1024


class LiveStreamHandler(Protocol):
    """Receives decoded messages. Calls are serialized: never two at once."""

    def on_connect(self, peer: str) -> None: ...

    def on_message(self, message: Message) -> None: ...

    def on_disconnect(self, peer: str, error: BaseException | None) -> None: ...


class _Outbox:
    """Per-connection sender: replaceable messages keep only the latest per key; others queue (bounded)."""

    def __init__(self, connection: socket.socket, max_queued: int = 64):
        self._connection = connection
        self._cond = threading.Condition()
        self._latest: dict[str, bytes] = {}
        self._queue: list[bytes] = []
        self._max_queued = max_queued
        self._closed = False
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="mq3drecon-live-outbox", daemon=True)
        self._thread.start()

    def put(self, data: bytes, replace_key: str | None) -> None:
        with self._cond:
            if self._closed:
                return
            if replace_key is not None:
                if replace_key in self._latest:
                    self.dropped += 1
                self._latest[replace_key] = data
            else:
                if len(self._queue) >= self._max_queued:
                    self._queue.pop(0)
                    self.dropped += 1
                self._queue.append(data)
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._closed and not self._queue and not self._latest:
                    self._cond.wait()
                if self._closed:
                    return
                if self._queue:
                    data = self._queue.pop(0)
                else:
                    key = next(iter(self._latest))
                    data = self._latest.pop(key)
            try:
                self._connection.sendall(data)
            except OSError:
                return


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
        self._outbox: _Outbox | None = None
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

    def send(self, message: Message, *, replace_key: str | None = None) -> bool:
        """Queue a message to the connected headset. ``replace_key``: keep only the newest message with that key.

        Never blocks the caller. Returns False when no headset is connected.
        """
        with self._state_lock:
            outbox = self._outbox
        if outbox is None:
            return False
        outbox.put(encode_message(message), replace_key)
        return True

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._connection is not None

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
            self._outbox = _Outbox(connection)
            self._reader_thread = threading.Thread(
                target=self._read_loop, args=(connection, peer), name="mq3drecon-live-reader", daemon=True
            )
            self._reader_thread.start()

    def _close_current_connection(self) -> None:
        with self._state_lock:
            connection, thread, outbox = self._connection, self._reader_thread, self._outbox
            self._connection, self._reader_thread, self._outbox = None, None, None
        if outbox is not None:
            outbox.close()
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
