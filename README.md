# tmdiff

**`git diff` for your chat exports, server logs, and notes — 100% offline, zero LLMs.**

`tmdiff` filters out repetitive background noise, API secrets, and boilerplate from chat and log exports to isolate key turning points and causal links.

## Pipeline

1. **Streaming Harmonic Squeeze & Secret Scrub (`squeeze.py`)**: 4096-dim FNV-1a NLMS-Huber filter that collapses repetitive log lines in `O(1)` memory while redacting API keys (`sk-`, `ghp_`, `AKIA`, JWTs).
2. **True Timeline Anchoring (`timeline.py`)**: Extracts microsecond birth timestamps from platform IDs (Slack, Discord/X snowflakes, Gmail, UUIDv1/v7) and flags undecodable IDs as `UNANCHORED`.
3. **Structural Fingerprinting (`fingerprint.py`)**: Strips variable literals, URLs, and hex hashes while binding negations (`not converge` -> `¬converge`).
4. **Bidirectional Causal Turning Points (`causal.py`)**: Forward and retrocausal error learning (`√(F · Bᵀ)`) isolates high-surprise pivots and symmetric concept links.
5. **Lossless Transport & Local HTTP Proxy (`transport.py`, `server.py`)**: Packs verified `.utf8.json` envelopes or serves `/squeeze` and `/causal` over HTTP (`tmdiff serve 8787`).

## Quick Start

```bash
pip install tmdiff
cat app.log | tmdiff
tmdiff examples/sample_archive.jsonl --pack examples/out.utf8.json
tmdiff serve 8787
```

## Run Tests

```bash
python3 -m unittest discover -v tests
```

## License

MIT
