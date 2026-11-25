import unittest
from tmdiff.timeline import decode_id, anchor_event, UndecodableID
from tmdiff.transport import pack_events, unpack_events
from tmdiff.fingerprint import structural_hash, tokenize
from tmdiff.causal import CausalOrganism
from tmdiff.squeeze import NlmsHuberFilter


class TestTmdiffPipeline(unittest.TestCase):
    def test_decodable_ids(self):
        self.assertEqual(decode_id("slack", "1761930161.842119").year, 2025)
        self.assertEqual(decode_id("slack", "p1761930161842119").year, 2025)
        self.assertEqual(decode_id("twitter", 1585841080431321088).year, 2022)
        self.assertEqual(decode_id("discord", 175928847299117063).year, 2016)
        self.assertEqual(decode_id("gmail", "1a03e3779c43f667").year, 2026)
        self.assertEqual(
            decode_id("uuid", "018f6b1c-7a00-7000-8000-000000000000").year, 2024
        )

    def test_honest_refusal(self):
        with self.assertRaises(UndecodableID):
            decode_id("uuid", "509a9263-2bf0-421b-9853-1234567890ab")
        with self.assertRaises(UndecodableID):
            decode_id("reddit", "1a2b3c")
        with self.assertRaises(UndecodableID):
            decode_id("twitter", "12345")

    def test_anchor_and_roundtrip_transport(self):
        raw = [
            {"platform": "slack", "id": "1761930161.842119", "text": "hello world"},
            {"platform": "reddit", "id": "abc12", "text": "no timestamp here"},
        ]
        anchored = [anchor_event(e) for e in raw]
        first, second = anchored
        self.assertTrue(first["anchored"])
        self.assertFalse(second["anchored"])

        env = pack_events(anchored, source="test.jsonl")
        restored = unpack_events(env)
        self.assertEqual(restored, anchored)

    def test_structural_hash_and_negation(self):
        h1 = structural_hash("Signed-off-by: Abhi\nNode 42 failed at 1761930161")
        h2 = structural_hash("Signed-off-by: Space\nNode 99 failed at 1761939999")
        self.assertEqual(h1, h2)
        self.assertIn("¬converge", tokenize("solver did not converge"))

    def test_causal_turning_points(self):
        events = [
            {"text": "kubernetes node scheduling cpu memory check"},
            {"text": "kubernetes node scheduling cpu memory check"},
            {"text": "kubernetes node scheduling cpu memory check"},
            {"text": "riemann zeta zeros off-line detection threshold logarithm"},
            {"text": "kubernetes node scheduling cpu memory check"},
        ]
        org = CausalOrganism()
        scored = org.fit_and_score(events)
        pivot = next(row for row in scored if "riemann" in row["text"])
        self.assertTrue(pivot["turning_point"])

    def test_nlms_squeeze_and_secret_scrub(self):
        lines = []
        for i in range(1, 201):
            if i == 100:
                lines.append("2026-11-26T00:12:01Z ERROR db_pool: refused token=sk-live9876543210abcdef123456")
            else:
                lines.append(f"2026-11-26T00:{i%60:02d}:00Z INFO healthcheck worker-{i%8:02d} status=200")
        flt = NlmsHuberFilter()
        out = flt.squeeze_stream(lines)
        self.assertEqual(flt.stats["bursts"], 1)
        self.assertEqual(flt.stats["secrets"], 1)
        anomaly = next(e for e in out if e["kind"] == "anomaly")
        self.assertIn("[REDACTED_SECRET]", anomaly["text"])
        self.assertNotIn("sk-live", anomaly["text"])


if __name__ == "__main__":
    unittest.main()
