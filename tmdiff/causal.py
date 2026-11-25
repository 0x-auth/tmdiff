"""Stage 4: Forward & retrocausal error-driven learning (√(F · Bᵀ)) + surprise detection."""
import math
import numpy as np
from tmdiff.fingerprint import tokenize


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


class CausalOrganism:
    def __init__(self, max_vocab: int = 2000, lr: float = 0.08, decay: float = 0.5):
        self.max_vocab = max_vocab
        self.lr = lr
        self.decay = decay
        self.vocab: dict[str, int] = {}
        self.inv_vocab: list[str] = []
        self.W_fwd: np.ndarray | None = None
        self.W_bwd: np.ndarray | None = None

    def _build_vocab(self, tokenized_events: list[list[str]], max_df_ratio: float = 0.35):
        n_docs = max(1, len(tokenized_events))
        df: dict[str, int] = {}
        tf: dict[str, int] = {}
        for toks in tokenized_events:
            seen = set()
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
                if t not in seen:
                    df[t] = df.get(t, 0) + 1
                    seen.add(t)

        # Filter out high-frequency boilerplate (> max_df_ratio) unless archive is tiny
        filtered = [
            t for t, freq in tf.items()
            if (df[t] / n_docs) <= max_df_ratio or n_docs < 10
        ]
        filtered.sort(key=lambda t: (-tf[t], t))
        selected = filtered[: self.max_vocab]
        self.vocab = {t: i for i, t in enumerate(selected)}
        self.inv_vocab = selected
        V = len(selected)
        # Born with W = 2I ("what was just said tends to recur")
        self.W_fwd = 2.0 * np.eye(V, dtype=np.float64)
        self.W_bwd = 2.0 * np.eye(V, dtype=np.float64)

    def _vec(self, toks: list[str]) -> np.ndarray:
        V = len(self.vocab)
        v = np.zeros(V, dtype=np.float64)
        for t in toks:
            idx = self.vocab.get(t)
            if idx is not None:
                v[idx] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _run_pass(self, vectors: list[np.ndarray], W: np.ndarray) -> list[float]:
        V = W.shape[0]
        ctx = np.zeros(V, dtype=np.float64)
        surprises = []
        for y in vectors:
            if np.sum(y) == 0:
                surprises.append(0.0)
                ctx *= self.decay
                continue
            y_dist = y / np.sum(y)
            p = _softmax(W.T @ ctx)
            # Cross-entropy surprise in bits
            bits = -float(np.sum(y_dist * np.log2(np.clip(p, 1e-12, 1.0))))
            surprises.append(bits)
            # Update solely from prediction error
            W += self.lr * np.outer(ctx, (y_dist - p))
            ctx = self.decay * ctx + y
        return surprises

    def fit_and_score(self, events: list[dict]) -> list[dict]:
        tokenized = [tokenize(e.get("text", "")) for e in events]
        self._build_vocab(tokenized)
        V = len(self.vocab)
        if V == 0:
            return [dict(e, surprise_bits=0.0, turning_point=False) for e in events]

        vecs = [self._vec(toks) for toks in tokenized]
        fwd_bits = self._run_pass(vecs, self.W_fwd)
        bwd_bits = list(reversed(self._run_pass(list(reversed(vecs)), self.W_bwd)))

        combined = [math.sqrt(max(0.0, f * b)) for f, b in zip(fwd_bits, bwd_bits)]
        mean_s = sum(combined) / max(1, len(combined))
        std_s = math.sqrt(sum((x - mean_s) ** 2 for x in combined) / max(1, len(combined)))
        threshold = mean_s + 0.75 * std_s

        scored = []
        for ev, f_b, b_b, c_b in zip(events, fwd_bits, bwd_bits, combined):
            row = dict(ev)
            row["fwd_bits"] = round(f_b, 3)
            row["bwd_bits"] = round(b_b, 3)
            row["surprise_bits"] = round(c_b, 3)
            row["turning_point"] = bool(c_b > threshold and c_b > 0)
            scored.append(row)
        return scored

    def symmetric_links(self, top_k: int = 10) -> list[tuple[str, str, float]]:
        if self.W_fwd is None or self.W_bwd is None or len(self.inv_vocab) == 0:
            return []
        F = np.maximum(0.0, self.W_fwd - 2.0 * np.eye(len(self.inv_vocab)))
        B = np.maximum(0.0, self.W_bwd - 2.0 * np.eye(len(self.inv_vocab)))
        sym = np.sqrt(F * B.T)
        np.fill_diagonal(sym, 0.0)
        pairs = []
        V = len(self.inv_vocab)
        for i in range(V):
            for j in range(V):
                if sym[i, j] > 0:
                    pairs.append((self.inv_vocab[i], self.inv_vocab[j], round(float(sym[i, j]), 4)))
        pairs.sort(key=lambda x: -x[2])
        return pairs[:top_k]
