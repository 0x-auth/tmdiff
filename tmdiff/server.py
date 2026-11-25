"""Zero-dependency HTTP server mode (`tmdiff serve 8787`) for streaming squeeze & causal diff."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from tmdiff.squeeze import NlmsHuberFilter
from tmdiff.timeline import anchor_event
from tmdiff.fingerprint import structural_hash
from tmdiff.causal import CausalOrganism


class TmdiffHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "tmdiff", "version": "0.2.0"})
        else:
            self._send_json(404, {"error": "POST /squeeze or POST /causal"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8", errors="replace")

        if self.path == "/squeeze":
            lines = [l for l in raw_body.splitlines() if l.strip()]
            flt = NlmsHuberFilter()
            events = flt.squeeze_stream(lines)
            self._send_json(200, {"stats": flt.stats, "events": events})
            return

        if self.path == "/causal":
            raw_events = [json.loads(l) for l in raw_body.splitlines() if l.strip()]
            enriched = []
            for ev in raw_events:
                e = anchor_event(ev)
                e["struct_hash"] = structural_hash(e.get("text", ""))
                enriched.append(e)
            enriched.sort(key=lambda x: (0, x["timestamp"]) if x["anchored"] else (1, ""))
            org = CausalOrganism()
            scored = org.fit_and_score(enriched)
            self._send_json(
                200,
                {
                    "turning_points": [r for r in scored if r["turning_point"]],
                    "links": org.symmetric_links(top_k=10),
                    "events": scored,
                },
            )
            return

        self._send_json(404, {"error": "Unknown endpoint; use /squeeze or /causal"})

    def log_message(self, format, *args):
        return


def run_server(port: int = 8787, host: str = "127.0.0.1"):
    srv = HTTPServer((host, port), TmdiffHandler)
    print(f"tmdiff listening on http://{host}:{port} (POST /squeeze, POST /causal, GET /health)")
    srv.serve_forever()
