# tmdiff

**`git diff` for your chat exports, logs, and notes — 100% offline, zero LLMs.**

When you export years of messages from Slack, Discord, Gmail, or AI chats, 95% of the file is repetitive noise, boilerplate, and broken timestamps. `tmdiff` filters out the predictable background and hands you back **only the turning points** where a new idea, bug, or shift actually happened.

## What It Does in 4 Steps

1. **Rebuilds the True Timeline (`timeline.py`)**
   Extracts exact microsecond creation timestamps straight from platform IDs (Slack `ts`, Discord/X snowflakes, Gmail hex IDs, UUIDv1/v7). If an ID has no timestamp (like UUIDv4 or Reddit), it marks it `UNANCHORED` instead of guessing.
2. **Strips Boilerplate & Variables (`fingerprint.py`)**
   Ignores numbers, URLs, hex hashes, and commit/export boilerplate (`Signed-off-by`, etc.), and binds negations (`not converge` → `¬converge`) so opposites don't look identical.
3. **Finds the Turning Points (`causal.py`)**
   Runs a lightweight forward-and-backward prediction model (`numpy` only) over your timeline. Routine messages score low surprise and get ignored; sudden conceptual shifts score high surprise (`bits`) and get flagged as **turning points**.
4. **Packs Lossless Handoffs (`transport.py`)**
   Optionally exports the cleaned timeline into a self-describing, round-trip-verified UTF-8 byte envelope (`.utf8.json`) that never breaks on quotes or special characters.

## Quick Start

```bash
pip install -e .
python3 -m tmdiff.cli examples/sample_archive.jsonl
```

### Example Output
```text
Processed 6 events | Turning points isolated:
  [2025-10-31T17:15:00.500000+00:00] (5.911 bits) Solver did not converge when mirror partner was omitted from state
  [2026-08-26T13:16:58.180000+00:00] (5.842 bits) Verified Davenport-Heilbronn mod 5 crossing at n=285 without Euler product
```

## Run Tests

```bash
python3 -m unittest discover -v tests
```

## License

MIT
