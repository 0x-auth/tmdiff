"""Stage 1: Recover exact birth timestamps from platform IDs and URLs without guessing."""
from datetime import datetime, timezone
import re
import uuid

TWITTER_EPOCH_MS = 1288834974657
DISCORD_EPOCH_MS = 1420070400000
UUID_EPOCH_OFFSET_100NS = 0x01B21DD213814000  # 1582-10-15 to 1970-01-01


class UndecodableID(ValueError):
    """Raised when an identifier or URL carries no embedded timestamp."""


def _extract_from_url(platform: str, url: str) -> tuple[str, str]:
    """Pull canonical platform and ID out of full Slack, Twitter/X, or Discord URLs."""
    m_slack = re.search(r"slack\.com/archives/[A-Z0-9]+/(p\d{16,})", url, re.IGNORECASE)
    if m_slack:
        return "slack", m_slack.group(1)

    m_tw = re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d{15,})", url, re.IGNORECASE)
    if m_tw:
        return "twitter", m_tw.group(1)

    m_dc = re.search(r"discord(?:app)?\.com/channels/\d+/\d+/(\d{15,})", url, re.IGNORECASE)
    if m_dc:
        return "discord", m_dc.group(1)

    return platform, url


def decode_id(platform: str, raw_id: str | int) -> datetime:
    p = platform.strip().lower()
    s = str(raw_id).strip()

    if s.startswith(("http://", "https://")):
        p, s = _extract_from_url(p, s)

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
    """Attach ISO timestamp if decodable from ID or URL; otherwise mark unanchored."""
    out = dict(event)
    try:
        dt = decode_id(event.get("platform", ""), event.get("id", ""))
        out["timestamp"] = dt.isoformat()
        out["anchored"] = True
    except (UndecodableID, ValueError):
        out["timestamp"] = None
        out["anchored"] = False
    return out
