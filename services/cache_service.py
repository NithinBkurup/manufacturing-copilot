"""
Cache Service – Manufacturing Copilot
Manages the OPC tag cache. Refreshed every 5 seconds by the cache loop.
All consumers read from this in-memory store — direct OPC polling per
query is prohibited.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("copilot.cache")


class CacheService:
    """Thread-safe in-memory OPC tag cache."""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Write (called only by opc_service cache loop)
    # ------------------------------------------------------------------

    async def update(self, tag: str, value: Any) -> None:
        async with self._lock:
            self._store[tag] = value
            self._timestamps[tag] = datetime.now()

    async def bulk_update(self, values: Dict[str, Any]) -> None:
        async with self._lock:
            now = datetime.now()
            for tag, value in values.items():
                self._store[tag] = value
                self._timestamps[tag] = now

    # ------------------------------------------------------------------
    # Read (called by manufacturing_service / ai context builder)
    # ------------------------------------------------------------------

    async def get(self, tag: str) -> Optional[Any]:
        async with self._lock:
            return self._store.get(tag)

    async def get_with_timestamp(self, tag: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if tag not in self._store:
                return None
            return {
                "value": self._store[tag],
                "timestamp": self._timestamps[tag].isoformat(),
                "age_sec": (datetime.now() - self._timestamps[tag]).total_seconds(),
            }

    async def get_all(self) -> Dict[str, Any]:
        async with self._lock:
            return dict(self._store)

    async def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return full cache snapshot with timestamps — for AI context injection."""
        async with self._lock:
            result = {}
            now = datetime.now()
            for tag, value in self._store.items():
                ts = self._timestamps.get(tag)
                result[tag] = {
                    "value": value,
                    "timestamp": ts.isoformat() if ts else None,
                    "age_sec": round((now - ts).total_seconds(), 1) if ts else None,
                }
            return result

    def is_stale(self, tag: str, max_age_sec: float = 30.0) -> bool:
        """Returns True if tag hasn't been updated within max_age_sec."""
        ts = self._timestamps.get(tag)
        if ts is None:
            return True
        return (datetime.now() - ts).total_seconds() > max_age_sec

    @property
    def tag_count(self) -> int:
        return len(self._store)

    @property
    def last_update(self) -> Optional[datetime]:
        if not self._timestamps:
            return None
        return max(self._timestamps.values())


# Module-level singleton — shared across the application
cache = CacheService()