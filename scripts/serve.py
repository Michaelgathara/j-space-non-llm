"""Launch the local horizon-lens viewer.

Usage:
    python scripts/serve.py --run-dir runs/default [--port 8731]

Then open http://127.0.0.1:8731/ — type a prompt, pick blocks and a horizon,
and read the recurrent workspace live.
"""

from __future__ import annotations

import argparse

from jspace.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8731)
    args = parser.parse_args()
    serve(args.run_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
