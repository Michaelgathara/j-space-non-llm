"""Local interactive viewer: type a prompt, read the workspace live.

Loads the checkpoint and fitted lens once (GPU if available), then serves a
single-page viewer on localhost. A readout request is one LSTM pass over the
prompt plus a handful of small matmuls — milliseconds — so the page feels
instant. Stdlib-only (``http.server``); binds 127.0.0.1 by default and is not
meant to face a network.

Routes:
    GET  /          the viewer page
    POST /api/grid  {prompt, blocks, max_horizon, min_token_count} -> grid JSON
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from torch import Tensor

from jspace.config import Paths
from jspace.data import Corpus, Vocab, tokenize
from jspace.lens import HorizonLens
from jspace.model import LSTMLanguageModel, load_checkpoint
from jspace.viz import build_grid_data, render_app_shell

DEFAULT_PROMPT = (
    "BAPTISTA : It likes me well . Biondello , hie you home , "
    "And bid Bianca make her ready straight ;"
)
ALL_BLOCKS = ["state", "h_top", "c1", "h0", "c0"]
MAX_PROMPT_TOKENS = 256


@dataclass
class LensApp:
    """The loaded model + lens, independent of HTTP (testable directly)."""

    model: LSTMLanguageModel
    lens: HorizonLens
    vocab: Vocab
    token_counts: Tensor
    device: str

    @classmethod
    def from_run_dir(cls, run_dir: str, device: str | None = None) -> LensApp:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = Paths(run_dir=run_dir)
        corpus = Corpus.load(paths.corpus_file)
        model = load_checkpoint(paths.checkpoint, device)
        lens = HorizonLens.load(paths.lens_file, device=device)
        lens.transports = lens.transports.to(device)
        return cls(
            model=model,
            lens=lens,
            vocab=corpus.vocab,
            token_counts=torch.bincount(corpus.train_ids, minlength=len(corpus.vocab)),
            device=device,
        )

    def config(self) -> dict:
        return {
            "model": self.lens.model_config,
            "max_horizon": self.lens.max_horizon,
            "blocks": ALL_BLOCKS,
            "default_blocks": ["state", "h_top", "c1"],
            "default_horizon": min(8, self.lens.max_horizon),
            "default_min_token_count": 5,
            "default_prompt": DEFAULT_PROMPT,
            "device": self.device,
        }

    def grid(self, params: dict) -> dict:
        """Validate request params and compute grid data. Raises ValueError
        with a user-facing message on bad input."""
        words = tokenize(str(params.get("prompt", "")))
        if not words:
            raise ValueError("prompt is empty")
        if len(words) > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt too long ({len(words)} > {MAX_PROMPT_TOKENS} tokens)")

        blocks = [b for b in params.get("blocks", []) if b in ALL_BLOCKS]
        if not blocks:
            raise ValueError("no valid blocks selected")

        max_horizon = int(params.get("max_horizon", 8))
        if not 1 <= max_horizon <= self.lens.max_horizon:
            raise ValueError(f"max_horizon must be in [1, {self.lens.max_horizon}]")
        min_count = max(0, int(params.get("min_token_count", 0)))

        token_ids = self.vocab.encode(words).to(self.device)
        return build_grid_data(
            self.model,
            self.lens,
            self.vocab,
            token_ids,
            blocks,
            max_horizon,
            min_token_count=min_count,
            token_counts=self.token_counts,
        )


def _make_handler(app: LensApp) -> type[BaseHTTPRequestHandler]:
    shell = render_app_shell(app.config()).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

        def do_GET(self) -> None:  # http.server API name
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", shell)
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # http.server API name
            if self.path != "/api/grid":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                params = json.loads(self.rfile.read(length) or b"{}")
                self._send_json(200, app.grid(params))
            except ValueError as err:
                self._send_json(400, {"error": str(err)})
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON body"})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[serve] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def serve(run_dir: str, host: str = "127.0.0.1", port: int = 8731) -> None:
    app = LensApp.from_run_dir(run_dir)
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    print(f"[serve] model+lens loaded on {app.device}")
    print(f"[serve] viewer at http://{host}:{port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
