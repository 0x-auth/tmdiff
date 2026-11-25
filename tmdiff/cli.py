"""CLI entry point for tmdiff."""
import argparse
import json
from tmdiff.timeline import anchor_event
from tmdiff.fingerprint import structural_hash
from tmdiff.causal import CausalOrganism
from tmdiff.transport import pack_events


def main():
    parser = argparse.ArgumentParser(prog="tmdiff", description="Causal & structural archive diff")
    parser.add_argument("input", help="Path to JSONL file of {platform, id, text} events")
    parser.add_argument("--pack", help="Optional path to write round-trip verified .utf8.json envelope")
    parser.add_argument("--links", type=int, default=8, help="Number of symmetric causal links to display")
    args = parser.parse_args()

    raw_events = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_events.append(json.loads(line))

    enriched = []
    seen_hashes = set()
    for ev in raw_events:
        e = anchor_event(ev)
        h = structural_hash(e.get("text", ""))
        e["struct_hash"] = h
        e["duplicate_structure"] = h in seen_hashes
        seen_hashes.add(h)
        enriched.append(e)

    # Sort anchored events chronologically while preserving relative order of unanchored ones
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
