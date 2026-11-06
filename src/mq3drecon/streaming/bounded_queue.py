"""A thread-safe bounded queue that drops the oldest item when full."""

from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """Bounded FIFO. ``put`` never blocks; when full, the oldest item is discarded and counted."""

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._items: deque[T] = deque()
        self._max_size = max_size
        self._cond = threading.Condition()
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)

    def put(self, item: T) -> bool:
        """Add an item. Returns ``False`` when an older item had to be dropped to make room."""
        with self._cond:
            if self._closed:
                return False
            dropped = False
            if len(self._items) >= self._max_size:
                self._items.popleft()
                self._dropped += 1
                dropped = True
            self._items.append(item)
            self._cond.notify()
            return not dropped

    def get(self, timeout: float | None = None) -> T | None:
        """Pop the oldest item, waiting up to ``timeout`` seconds. Returns ``None`` on timeout or close."""
        with self._cond:
            if not self._items and not self._closed:
                self._cond.wait(timeout=timeout)
            if not self._items:
                return None
            return self._items.popleft()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
