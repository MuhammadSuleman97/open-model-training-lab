#!/usr/bin/env python3
"""Serve the self-contained Open Model Training Lab learning guide."""

from __future__ import annotations

import argparse
import functools
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEARNING_DIR = PROJECT_ROOT / "learning"
REQUIRED_FILES = ("index.html", "styles.css", "app.js")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8090)
parser.add_argument("--no-browser", action="store_true")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()

if args.host not in {"127.0.0.1", "localhost", "::1"}:
    raise SystemExit("Learning lab refused: host must be a loopback address.")
if not 1 <= args.port <= 65535:
    raise SystemExit("Learning lab refused: port must be between 1 and 65535.")

missing = [name for name in REQUIRED_FILES if not (LEARNING_DIR / name).is_file()]
if missing:
    raise SystemExit(f"Learning lab failed: missing files: {', '.join(missing)}")

if args.check:
    print("Open Model Training Lab — learning guide preflight")
    print(f"learning_dir: {LEARNING_DIR}")
    print(f"files: {', '.join(REQUIRED_FILES)}")
    print("learning_guide_preflight_ok: True")
    raise SystemExit(0)

handler = functools.partial(SimpleHTTPRequestHandler, directory=str(LEARNING_DIR))
server = HTTPServer((args.host, args.port), handler)
url = f"http://{args.host}:{args.port}/"

print("Open Model Training Lab — interactive learning guide", flush=True)
print(f"learning_guide_ready: {url}", flush=True)
print("stop_with: Control-C", flush=True)
if not args.no_browser:
    threading.Timer(0.35, webbrowser.open, args=(url,)).start()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nlearning_guide_stopped: True", flush=True)
finally:
    server.server_close()
