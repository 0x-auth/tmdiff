"""Stage 1: Recover exact birth timestamps from platform IDs without guessing."""
from datetime import datetime, timezone
from typing import Optional
import uuid

TWITTER_EPOCH_MS = 1288834974657
DISCORD_EPOCH_MS = 1420070400000
UUID_EPOCH_OFFSET_100NS = 0x01B21DD213814000  # 1582-10-15 to 1970-01-01


class UndecodableID(ValueError):
    """Raised when an identifier carries no embedded timestamp."""


def decode_id(platform: str, raw_id: str | int) -> datetime:
    p = platform.strip().lower()
    s = str(raw_id).strip()

    if p == "slack":
        if s.startswith("p") and s[1:].isdigit() and len(s) >= 17:
            sec = int(s[1:11])
            usec = int(s[11:17])
            return datetime.fromtimestamp(sec + usec / 1_000_000, tz=timezone.utc)
        if "." in s:
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        raise UndecodableID(f"Slack ID lacks microsecond ts: {s}")

    if p in ("twitter", "x"):
        if not s.isdigit() or len(s) < 15:
            raise UndecodableID(f"Sequential/pre-snowflake Twitter ID: {s}")
        ms = (int(s) >> 22) + TWITTER_EPOCH_MS
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

    if p == "discord":
        if not s.isdigit():
            raise UndecodableID(f"Invalid Discord snowflake: {s}")
        ms = (int(s) >> 22) + DISCORD_EPOCH_MS
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

    if p == "gmail":
        if len(s) <= 5 or not all(c in "0123456789abcdefABCDEF" for c in s):
            raise UndecodableID(f"Invalid Gmail hex ID: {s}")
        ms = int(s[:-5], 16)
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

    if p == "uuid":
        u = uuid.UUID(s)
        if u.version == 7:
            ms = int.from_bytes(u.bytes[:6], "big")
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        if u.version == 1:
            ts_100ns = u.time - UUID_EPOCH_OFFSET_100NS
            return datetime.fromtimestamp(ts_100ns / 10_000_000.0, tz=timezone.utc)
        raise UndecodableID(f"UUIDv{u.version} carries no timestamp: {s}")

    raise UndecodableID(f"Unsupported or unanchored platform: {platform}")


def anchor_event(event: dict) -> dict:
    """Attach ISO timestamp if decodable; otherwise explicitly mark unanchored."""
    out = dict(event)
    try:
        dt = decode_id(event.get("platform", ""), event.get("id", ""))
        out["timestamp"] = dt.isoformat()
        out["anchored"] = True
    except (UndecodableID, ValueError):
        out["timestamp"] = None
        out["anchored"] = False
    return out
