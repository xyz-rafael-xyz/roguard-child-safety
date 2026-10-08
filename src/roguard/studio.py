"""Loopback-only browser workspace for declared, read-only contract checks."""

from __future__ import annotations

import argparse
import hmac
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources

from .cli import _strict_json_loads, assess_json
from .contrast import contrast_declared_contracts
from .explore import explore_declared_contract
from .v1_contract import assess_v1_json
from .v1_explore import explore_v1_contract

MAX_REQUEST_BYTES = 1_048_576
ASSETS = {"/": ("studio.html", "text/html; charset=utf-8"),
          "/studio.js": ("studio.js", "text/javascript; charset=utf-8"),
          "/studio.css": ("studio.css", "text/css; charset=utf-8")}
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self'; "
                                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
}


def make_server(port: int = 0) -> HTTPServer:
    """Create an ephemeral loopback service; caller owns serve_forever/shutdown."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be from 0 to 65535")
    token = secrets.token_hex(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass  # Inputs and paths are never logged.

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _trusted_host(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self) -> None:
            if not self._trusted_host():
                self._json(403, {"error": "Forbidden host"})
                return
            asset = ASSETS.get(self.path)
            if asset is None:
                self._json(404, {"error": "Not found"})
                return
            name, content_type = asset
            data = resources.read_binary("roguard", name)
            if name == "studio.html":
                data = data.replace(b"__ROGUARD_TOKEN__", token.encode("ascii"))
            self._send(200, data, content_type)

        def do_POST(self) -> None:
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (not self._trusted_host() or self.headers.get("Origin") != origin or
                    not hmac.compare_digest(self.headers.get("X-RoGuard-Token", ""), token)):
                self._json(403, {"error": "Forbidden request"})
                return
            if self.path not in {"/api/assess", "/api/explore", "/api/contrast",
                                 "/api/v1/assess", "/api/v1/explore"}:
                self._json(404, {"error": "Not found"})
                return
            if self.headers.get("Content-Type") != "application/json" or self.headers.get("Transfer-Encoding"):
                self._json(415, {"error": "JSON body required"})
                return
            length = self.headers.get("Content-Length", "")
            if not length.isdecimal():
                self._json(400, {"error": "Content length required"})
                return
            size = int(length)
            if size > MAX_REQUEST_BYTES:
                self._json(413, {"error": "Request exceeds the local size limit"})
                return
            if size == 0:
                self._json(400, {"error": "Empty request"})
                return
            try:
                payload = _strict_json_loads(self.rfile.read(size).decode("utf-8"))
                if self.path == "/api/v1/explore":
                    result = explore_v1_contract(payload)
                elif self.path == "/api/v1/assess":
                    result = assess_v1_json(payload)
                elif self.path == "/api/assess":
                    result = assess_json(payload)
                elif self.path == "/api/explore":
                    result = explore_declared_contract(payload)
                else:
                    if not isinstance(payload, dict) or set(payload) != {"before", "after"}:
                        raise ValueError("Contrast requires before and after")
                    result = contrast_declared_contracts(payload["before"], payload["after"])
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                self._json(400, {"error": "Invalid JSON or declared contract"})
                return
            self._json(200, result)

    return HTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a local, read-only RoGuard decision workspace")
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 chooses a free port")
    parser.add_argument("--open", action="store_true", help="Open the local page in your browser")
    args = parser.parse_args()
    try:
        server = make_server(args.port)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"RoGuard Decision Studio: {url}", flush=True)
    print("Runs on this computer only. Press Ctrl+C to stop.", flush=True)
    if args.open:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
