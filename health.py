"""Render 'Web Service' port talab qiladi + UptimeRobot uchun /health. Faqat standart kutubxona."""
from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_HEAD = do_GET  # UptimeRobot HEAD ham yuborishi mumkin

    def log_message(self, *args):  # jim
        pass


def start_health_server(port: int, host: str = "0.0.0.0") -> HTTPServer:
    server = HTTPServer((host, port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="health").start()
    logging.info("Health server %s:%s da ishga tushdi", host, port)
    return server
