"""After a live session ends: compact the recording in place (keep only frames worth keeping).

The compact copy is written next to the session first; only after it succeeds is
the full recording removed (``build.keep_full_recording`` keeps it as
``<name>_full``). Live outputs (``live_mesh*.ply``, ``live_session.json``) are carried over.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mq3drecon.config.live_config import LiveConfig

logger = logging.getLogger(__name__)

_CARRIED = ("live_session.json",)


def compact_in_place(session_dir: Path, config: LiveConfig) -> Path:
    from mq3drecon.processing.live.keyframes import KeyframePolicy
    from mq3drecon.workflows.compaction import compact_session

    from dataclasses import asdict

    session_dir = Path(session_dir)
    staging = session_dir.with_name(session_dir.name + ".compacting")
    if staging.exists():
        shutil.rmtree(staging)
    report = compact_session(
        session_dir, staging, policy=KeyframePolicy(**asdict(config.keyframes)), device=config.volume.device,
    )
    for path in session_dir.glob("live_mesh*.ply"):
        shutil.copy2(path, staging / path.name)
    for name in _CARRIED:
        if (session_dir / name).exists():
            shutil.copy2(session_dir / name, staging / name)
    if config.build.keep_full_recording:
        session_dir.rename(session_dir.with_name(session_dir.name + "_full"))
    else:
        shutil.rmtree(session_dir)
    staging.rename(session_dir)
    logger.info("Compacted %s: %.0f MB -> %.0f MB", session_dir, report.bytes_before / 1e6, report.bytes_after / 1e6)
    print(f"[live] Compacted {session_dir.name}: {report.bytes_before / 1e6:.0f} MB -> {report.bytes_after / 1e6:.0f} MB "
          f"({report.ratio:.0f}x), keyframes {report.frames_after}")
    return session_dir
