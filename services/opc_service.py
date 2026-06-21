"""
OPC Service – Manufacturing Copilot
Connects to Takebishi DXP OPC UA server (PNRMPAS_DXPOPCUA_LIVE.dxp).
Refreshes tag cache every 5 seconds via cache_service.py.

Rule: Direct OPC polling per user query is PROHIBITED.
      All OPC data is served from cache_service only.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from asyncua import Client as OPCClient
    from asyncua.common.node import Node
    OPC_AVAILABLE = True
except ImportError:
    OPC_AVAILABLE = False
    logging.getLogger("copilot.opc").warning(
        "asyncua not installed — OPC service running in simulation mode"
    )

from config.settings import settings
from services.cache_service import cache

logger = logging.getLogger("copilot.opc")


# ---------------------------------------------------------------------------
# Tag Map — Takebishi DXP Tag Addresses
# ---------------------------------------------------------------------------
# Update NodeIds to match your PNRMPAS_DXPOPCUA_LIVE.dxp configuration.
# Format: ns={namespace};s={tag_path}
#
# Expected data per spec:
#   - Live machine status per line
#   - Current recipe / program number running
#   - Production counters (shift actual)
#   - Process status (tool OK / NG)
# ---------------------------------------------------------------------------

NS = settings.OPC_NAMESPACE  # Namespace index from Takebishi DXP

TAG_MAP: Dict[str, str] = {
    # --- Line 1 ---
    "Line1.MachineStatus":       f"ns={NS};s=Line1.MachineStatus",
    "Line1.LineStatus":          f"ns={NS};s=Line1.LineStatus",
    "Line1.CurrentRecipe":       f"ns={NS};s=Line1.CurrentRecipe",
    "Line1.ProgramNumber":       f"ns={NS};s=Line1.ProgramNumber",
    "Line1.ShiftActualCount":    f"ns={NS};s=Line1.ShiftActualCount",
    "Line1.ShiftTargetCount":    f"ns={NS};s=Line1.ShiftTargetCount",
    "Line1.ToolStatus":          f"ns={NS};s=Line1.ToolStatus",       # OK / NG
    "Line1.CurrentOrderNo":      f"ns={NS};s=Line1.CurrentOrderNo",

    # --- Line 2 ---
    "Line2.MachineStatus":       f"ns={NS};s=Line2.MachineStatus",
    "Line2.LineStatus":          f"ns={NS};s=Line2.LineStatus",
    "Line2.CurrentRecipe":       f"ns={NS};s=Line2.CurrentRecipe",
    "Line2.ProgramNumber":       f"ns={NS};s=Line2.ProgramNumber",
    "Line2.ShiftActualCount":    f"ns={NS};s=Line2.ShiftActualCount",
    "Line2.ShiftTargetCount":    f"ns={NS};s=Line2.ShiftTargetCount",
    "Line2.ToolStatus":          f"ns={NS};s=Line2.ToolStatus",
    "Line2.CurrentOrderNo":      f"ns={NS};s=Line2.CurrentOrderNo",

    # --- Line 3 ---
    "Line3.MachineStatus":       f"ns={NS};s=Line3.MachineStatus",
    "Line3.LineStatus":          f"ns={NS};s=Line3.LineStatus",
    "Line3.CurrentRecipe":       f"ns={NS};s=Line3.CurrentRecipe",
    "Line3.ProgramNumber":       f"ns={NS};s=Line3.ProgramNumber",
    "Line3.ShiftActualCount":    f"ns={NS};s=Line3.ShiftActualCount",
    "Line3.ShiftTargetCount":    f"ns={NS};s=Line3.ShiftTargetCount",
    "Line3.ToolStatus":          f"ns={NS};s=Line3.ToolStatus",
    "Line3.CurrentOrderNo":      f"ns={NS};s=Line3.CurrentOrderNo",

    # --- Line 4 ---
    "Line4.MachineStatus":       f"ns={NS};s=Line4.MachineStatus",
    "Line4.LineStatus":          f"ns={NS};s=Line4.LineStatus",
    "Line4.CurrentRecipe":       f"ns={NS};s=Line4.CurrentRecipe",
    "Line4.ShiftActualCount":    f"ns={NS};s=Line4.ShiftActualCount",
    "Line4.ShiftTargetCount":    f"ns={NS};s=Line4.ShiftTargetCount",
    "Line4.ToolStatus":          f"ns={NS};s=Line4.ToolStatus",
    "Line4.CurrentOrderNo":      f"ns={NS};s=Line4.CurrentOrderNo",
}


class OPCService:
    """
    Polls Takebishi DXP OPC UA server and keeps cache_service updated.
    Consumers MUST read from cache — never call OPC directly.
    """

    def __init__(self):
        self._url = settings.OPC_SERVER_URL
        self._interval = settings.OPC_CACHE_INTERVAL_SEC
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._connected = False
        self._last_error: Optional[str] = None
        self._poll_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_cache_loop(self) -> None:
        if not OPC_AVAILABLE:
            logger.warning("OPC UA library not available — starting simulation mode")
            self._task = asyncio.create_task(self._simulation_loop())
            return

        self._running = True
        self._task = asyncio.create_task(self._cache_loop())
        logger.info("OPC cache loop started — polling %s every %ds", self._url, self._interval)

    async def stop_cache_loop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OPC cache loop stopped")

    # ------------------------------------------------------------------
    # Cache loop — runs every 5 seconds
    # ------------------------------------------------------------------

    async def _cache_loop(self) -> None:
        while self._running:
            try:
                await self._poll_and_update()
                self._poll_count += 1
            except Exception as exc:
                self._last_error = str(exc)
                self._connected = False
                logger.debug("OPC poll error (will retry): %s", exc)
            finally:
                await asyncio.sleep(self._interval)

    async def _poll_and_update(self) -> None:
        """Connect, read all tags, push to cache. One round-trip per interval."""
        async with OPCClient(url=self._url) as client:
            client.session_timeout = settings.OPC_CONNECTION_TIMEOUT * 1000
            self._connected = True

            tag_values: Dict[str, Any] = {}
            for friendly_name, node_id in TAG_MAP.items():
                try:
                    node = client.get_node(node_id)
                    value = await node.read_value()
                    tag_values[friendly_name] = value
                except Exception as exc:
                    logger.debug("Tag read error %s: %s", friendly_name, exc)
                    tag_values[friendly_name] = None

            await cache.bulk_update(tag_values)
            logger.debug("OPC cache updated — %d tags", len(tag_values))

    # ------------------------------------------------------------------
    # Simulation mode (when asyncua not installed / OPC server not available)
    # ------------------------------------------------------------------

    async def _simulation_loop(self) -> None:
        """Populate cache with realistic simulated data for development."""
        import random
        logger.info("OPC simulation mode active — cache populated with demo data")
        while True:
            sim_data = {}
            for line in range(1, 5):
                prefix = f"Line{line}"
                sim_data[f"{prefix}.MachineStatus"] = random.choice(["RUNNING", "RUNNING", "IDLE", "FAULT"])
                sim_data[f"{prefix}.LineStatus"]    = "AUTO"
                sim_data[f"{prefix}.CurrentRecipe"] = f"RCP-{random.randint(100, 999)}"
                sim_data[f"{prefix}.ProgramNumber"] = str(random.randint(1, 50))
                sim_data[f"{prefix}.ShiftActualCount"] = random.randint(80, 140)
                sim_data[f"{prefix}.ShiftTargetCount"] = 120
                sim_data[f"{prefix}.ToolStatus"]    = random.choice(["OK", "OK", "OK", "NG"])
                sim_data[f"{prefix}.CurrentOrderNo"] = str(random.randint(147000000, 148000000))
            await cache.bulk_update(sim_data)
            await asyncio.sleep(self._interval)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "opc_url": self._url,
            "poll_interval_sec": self._interval,
            "poll_count": self._poll_count,
            "last_error": self._last_error,
            "cache_tag_count": cache.tag_count,
            "cache_last_update": cache.last_update.isoformat() if cache.last_update else None,
        }