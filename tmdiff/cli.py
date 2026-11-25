"""Unified CLI for tmdiff: streaming sqz noise collapse + ID anchoring + causal diff."""
import argparse
import json
import sys
from tmdiff.timeline import anchor_event
from tmdiff.fingerprint import structural_hash
from tmdiff.causal import CausalOrganism
from tmdiff.squeeze import NlmsHuberFilter
from tmdiff.transport import pack_events
from tmdiff.server import run_server


def main():
    parser = argparse.ArgumentParser(
        prog="tmdiff",
        description="Streaming harmonic noise filter + ID timeline anchoring + causal archive diff.",
    )
    parser.add_argument("input", nargs="?", default="-", help="JSONL or raw log file, or 'serve'")
    parser.add_argument("port", nargs="?", type=int, default=8787, help="Port when running 'tmdiff serve 8787'")
    parser.add_argument("--mode", choices=["auto", "squeeze", "causal"], default="auto")
    parser.add_argument("--pack", help="Optional path to write round-trip verified .utf8.json envelope")
    parser.add_argument("--links", type=int, default=8, help="Number of symmetric causal links to show")
    args = parser.parse_args()

    if args.input == "serve":
        run_server(port=args.port)
        return

    fh = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    raw_lines = [line.rstrip("\r\n") for line in fh if line.strip()]
    if fh is not sys.stdin:
        fh.close()

    is_jsonl = raw_lines and next(iter(raw_lines)).lstrip().startswith("{")
    mode = ("causal" if is_jsonl else "squeeze") if args.mode == "auto" else args.mode

    if mode == "squeeze":
        flt = NlmsHuberFilter()
        events = flt.squeeze_stream(raw_lines)
        for ev in events:
            print(ev["text"])
        s = flt.stats
        pct = (1.0 - sum(len(e["text"]) + 1 for e in events) / max(1, s["in_b"])) * 100.0
        sys.stderr.write(
            f"\n[tmdiff squeeze] {s['in_l']} lines -> {len(events)} events "
            f"({max(0.0, pct):.2f}% noise collapsed | {s['bursts']} bursts | {s['secrets']} secrets scrubbed)\n"
        )
        if args.pack:
            env = pack_events(events, source=args.input)
            with open(args.pack, "w", encoding="utf-8") as f:
                json.dump(env, f, ensure_ascii=False, indent=2)
        return

    raw_events = [json.loads(line) for line in raw_lines]
    enriched = []
    seen_hashes = set()
    for ev in raw_events:
        e = anchor_event(ev)
        h = structural_hash(e.get("text", ""))
        e["struct_hash"] = h
        e["duplicate_structure"] = h in seen_hashes
        seen_hashes.add(h)
        enriched.append(e)

    enriched.sort(key=lambda x: (0, x["timestamp"]) if x["anchored"] else (1, ""))

    org = CausalOrganism()
    scored = org.fit_and_score(enriched)

    print(f"Processed {len(scored)} events | Turning points isolated:")
    for row in scored:
        if row["turning_point"]:
            ts = row["timestamp"] or "UNANCHORED"
            print(f"  [{ts}] ({row['surprise_bits']} bits) {row['text'][:72]}")

    print("\nTop symmetric causal links √(F·Bᵀ):")
    for a, b, score in org.symmetric_links(top_k=args.links):
        print(f"  {a} <-> {b} : {score}")

    if args.pack:
        env = pack_events(scored, source=args.input)
        with open(args.pack, "w", encoding="utf-8") as f:
            json.dump(env, f, ensure_ascii=False, indent=2)
        print(f"\nPacked verified UTF-8 envelope -> {args.pack}")


if __name__ == "__main__":
    main()
