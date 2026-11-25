"""Stage 2: Self-describing UTF-8 byte-list transport with round-trip verification."""
import json
import re

HOW = (
    "Each list in 'parts' is UTF-8 bytes. Decode with bytes(list).decode('utf-8'). "
    "'kind' specifies reassembly."
)
JOIN = {
    "text": "parts are paragraphs: join with \\n\\n",
    "jsonl": "parts are lines: join with \\n",
}


def pack_events(events: list[dict], source: str = "stream.jsonl") -> dict:
    lines = [json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in events]
    parts = [list(line.encode("utf-8")) for line in lines]
    envelope = {
        "format": "utf-8 byte lists",
        "kind": "jsonl",
        "join": JOIN["jsonl"],
        "how_to_decode": HOW,
        "source": source,
        "parts": parts,
    }
    unpacked = unpack_events(envelope)
    assert unpacked == events, "Round-trip verification failed"
    return envelope


def unpack_events(envelope: dict) -> list[dict]:
    if envelope.get("format") != "utf-8 byte lists":
        raise ValueError("Unsupported envelope format")
    strs = [bytes(p).decode("utf-8") for p in envelope["parts"]]
    if envelope.get("kind") == "jsonl":
        return [json.loads(s) for s in strs if s.strip()]
    if envelope.get("kind") == "text":
        return [{"text": "\n\n".join(strs)}]
    raise ValueError(f"Unsupported kind: {envelope.get('kind')}")
