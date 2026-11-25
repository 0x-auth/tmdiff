"""Stage 0/3: Streaming NLMS-Huber harmonic noise filter & secret scrubber (zero deps)."""
from collections import deque
import math

DIM = 4096
DIM_MASK = DIM - 1


def fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def structural_skeleton(line: str) -> str:
    out = []
    for tok in line.split():
        t = tok.strip(",;\"'")
        if not t:
            continue
        s = t[2:] if t.startswith("0x") else t
        if s and all(c in "0123456789abcdefABCDEF.:-" for c in s) and any(c.isdigit() for c in s):
            out.append("<NUM>")
        elif len(t) >= 32 and all(c in "0123456789abcdefABCDEF-" for c in t):
            out.append("<UUID>")
        else:
            out.append("".join("#" if c.isdigit() else c for c in t))
    return " ".join(out)


def _is_secret(val: str) -> bool:
    c = val.strip("\"',;")
    return (
        (c.startswith("sk-") and len(c) > 20)
        or (c.startswith("ghp_") and len(c) > 20)
        or (c.startswith("AKIA") and len(c) == 20)
        or (c.startswith("eyJ") and c.count(".") == 2 and len(c) > 40)
    )


def scrub_secrets(line: str) -> tuple[str, int]:
    count = 0
    words = []
    for w in line.split(" "):
        if "=" in w:
            k, v = w.split("=", 1)
            if _is_secret(v):
                count += 1
                words.append(f"{k}=[REDACTED_SECRET]")
                continue
        if _is_secret(w):
            count += 1
            words.append("[REDACTED_SECRET]")
        else:
            words.append(w)
    return " ".join(words), count


class NlmsHuberFilter:
    def __init__(self, mu: float = 0.45, delta: float = 0.45):
        self.weights = [0.0] * DIM
        self.prev_u: dict[int, float] = {}
        self.mu = mu
        self.delta = delta
        self.eps = 1e-6
        self.pre_ctx: deque[tuple[int, str]] = deque(maxlen=1)
        self.post_rem = 0
        self.collapsed_run = 0
        self.in_burst = False
        self.stats = dict(in_b=0, out_b=0, in_l=0, out_l=0, bursts=0, secrets=0)

    def vectorize(self, line: str) -> dict[int, float]:
        skel = structural_skeleton(line)
        b = skel.encode("utf-8")
        if not b:
            return {}
        counts: dict[int, float] = {}
        for tok in skel.split():
            h = fnv1a_64(tok.encode("utf-8"))
            idx = h & DIM_MASK
            sign = 1.0 if (h >> 63) == 0 else -1.0
            counts[idx] = counts.get(idx, 0.0) + sign * 1.5
        if len(b) >= 3:
            for i in range(len(b) - 2):
                h = fnv1a_64(b[i : i + 3])
                idx = h & DIM_MASK
                sign = 1.0 if (h >> 63) == 0 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
        norm_sq = sum(v * v for v in counts.values())
        if norm_sq <= self.eps:
            return {}
        inv = 1.0 / math.sqrt(norm_sq)
        return {idx: val * inv for idx, val in counts.items()}

    def step(self, line: str) -> tuple[bool, float]:
        u = self.vectorize(line)
        if not u:
            return False, 0.0
        y_hat = sum(self.weights[idx] * val for idx, val in u.items())
        rep_sim = sum(self.prev_u.get(idx, 0.0) * val for idx, val in u.items())
        self.prev_u = u

        effective_predict = max(y_hat, rep_sim * 0.95)
        err = max(0.0, 1.0 - effective_predict)

        adapt_rate = self.mu if (err <= self.delta or self.stats["in_l"] <= 5) else (self.mu * 0.25)
        step = (adapt_rate * max(0.0, 1.0 - y_hat)) / (1.0 + self.eps)
        for idx, val in u.items():
            self.weights[idx] = max(-1.5, min(1.5, self.weights[idx] + step * val))

        return (err > self.delta and self.stats["in_l"] > 2), round(err, 4)

    def squeeze_stream(self, lines) -> list[dict]:
        """Filter an iterable of raw lines into compact turning-point events with context."""
        emitted: list[dict] = []
        for raw in lines:
            raw = raw.rstrip("\r\n")
            self.stats["in_l"] += 1
            self.stats["in_b"] += len(raw) + 1
            clean, sec = scrub_secrets(raw)
            self.stats["secrets"] += sec
            is_anom, err = self.step(clean)
            idx = self.stats["in_l"]

            if not is_anom:
                if self.post_rem > 0:
                    emitted.append({"line": idx, "kind": "context", "error": err, "text": clean})
                    self.post_rem -= 1
                else:
                    self.collapsed_run += 1
                    self.in_burst = False
                    self.pre_ctx.append((idx, clean))
            else:
                if self.collapsed_run > 0:
                    emitted.append({
                        "line": idx,
                        "kind": "collapsed",
                        "collapsed_count": self.collapsed_run,
                        "text": f"··· [×{self.collapsed_run} harmonic lines collapsed] ···",
                    })
                    self.collapsed_run = 0
                if not self.in_burst:
                    self.stats["bursts"] += 1
                    self.in_burst = True
                    while self.pre_ctx:
                        p_idx, p_txt = self.pre_ctx.popleft()
                        emitted.append({"line": p_idx, "kind": "context", "error": 0.0, "text": p_txt})
                emitted.append({"line": idx, "kind": "anomaly", "error": err, "text": clean})
                self.post_rem = 1

        if self.collapsed_run > 0:
            emitted.append({
                "line": self.stats["in_l"],
                "kind": "collapsed",
                "collapsed_count": self.collapsed_run,
                "text": f"··· [×{self.collapsed_run} harmonic lines collapsed] ···",
            })
        return emitted
