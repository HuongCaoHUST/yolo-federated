import base64
import html
import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(os.environ.get("FL_CONFIG_PATH", BASE_DIR / "config_server.json"))
RUNS_DIR = Path(os.environ.get("FL_RUNS_DIR", BASE_DIR / "runs"))
FLOWER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080")
UI_ADDRESS = os.environ.get("FL_UI_ADDRESS", "0.0.0.0:5000")
ADMIN_USER = os.environ.get("FL_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("FL_ADMIN_PASSWORD", "change-me")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class RunManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.current_run = None
        self.log_handle = None

    def status(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            exit_code = None if self.process is None or running else self.process.returncode
            return {"running": running, "run": self.current_run, "exit_code": exit_code}

    def start(self, name, config):
        with self.lock:
            self.stop()
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip("-.")
            safe_name = safe_name or "experiment"
            run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{safe_name}"
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log_handle = (run_dir / "server.log").open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env.update({
                "FL_CONFIG_PATH": str(config_path),
                "FL_OUTPUT_DIR": str(run_dir),
                "FL_SERVER_ADDRESS": FLOWER_ADDRESS,
            })
            self.process = subprocess.Popen(
                ["python", "server/main.py"],
                cwd=BASE_DIR.parent,
                env=env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.current_run = run_id
            return run_id

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None

    def log(self, lines=200):
        with self.lock:
            if not self.current_run:
                return "Chưa có experiment nào được chạy."
            path = RUNS_DIR / self.current_run / "server.log"
            if not path.exists():
                return "Log chưa được tạo."
            return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def runs(self):
        result = []
        for path in sorted(RUNS_DIR.iterdir(), reverse=True):
            if path.is_dir():
                result.append({
                    "name": path.name,
                    "model": (path / "model_federated_final.pt").exists(),
                    "report": (path / "rapport_federated_learning.csv").exists(),
                })
        return result[:30]


manager = RunManager()


def load_default_config():
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def page(message=""):
    config = load_default_config()
    status = manager.status()
    runs = manager.runs()
    state = "ĐANG CHẠY" if status["running"] else "ĐÃ DỪNG"
    state_class = "running" if status["running"] else "stopped"
    rows = "".join(
        f"<tr><td>{html.escape(run['name'])}</td><td>{'✓' if run['report'] else '–'}</td>"
        f"<td>{'✓' if run['model'] else '–'}</td></tr>" for run in runs
    ) or "<tr><td colspan='3'>Chưa có experiment</td></tr>"
    client_json = html.escape(json.dumps(config.get("clients", {}), ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO Federated Learning</title>
<style>
:root{{--bg:#0b1020;--card:#151c31;--line:#29334e;--text:#eef2ff;--muted:#9ba7c4;--accent:#6d8cff;--danger:#ef5b67;--ok:#39c98a}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#090d19,#111936);color:var(--text);font:15px system-ui,sans-serif}}
.wrap{{max-width:1080px;margin:auto;padding:32px 18px}} h1{{margin:0 0 6px;font-size:28px}} .sub{{color:var(--muted);margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .card{{background:rgba(21,28,49,.94);border:1px solid var(--line);border-radius:14px;padding:20px}}
.wide{{grid-column:1/-1}} label{{display:block;color:var(--muted);margin:13px 0 6px}} input,textarea{{width:100%;background:#0c1222;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px}}
textarea{{min-height:230px;font:13px ui-monospace,monospace}} button{{border:0;border-radius:8px;padding:11px 16px;color:white;background:var(--accent);cursor:pointer;font-weight:650;margin:12px 8px 0 0}}
button.danger{{background:var(--danger)}} .badge{{display:inline-block;padding:6px 10px;border-radius:20px;font-weight:700}} .running{{background:#123e32;color:#66e4ae}} .stopped{{background:#49242a;color:#ff929b}}
pre{{background:#080d19;border:1px solid var(--line);border-radius:8px;padding:14px;max-height:420px;overflow:auto;white-space:pre-wrap;color:#c7d2ee}}
table{{width:100%;border-collapse:collapse}} td,th{{padding:9px;border-bottom:1px solid var(--line);text-align:left}} .msg{{background:#27375e;padding:10px;border-radius:8px;margin-bottom:14px}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<h1>YOLO Federated Learning</h1><div class="sub">Quản lý các lượt huấn luyện trên VPS</div>
{f'<div class="msg">{html.escape(message)}</div>' if message else ''}
<div class="grid"><section class="card"><h2>Trạng thái</h2><span class="badge {state_class}">{state}</span>
<p>Experiment: <b>{html.escape(status['run'] or '–')}</b></p>
<form method="post" action="/stop"><button class="danger">Dừng server</button></form></section>
<section class="card"><h2>Tạo lượt chạy mới</h2><form method="post" action="/start">
<label>Tên test</label><input name="name" value="test" required>
<label>Số round</label><input type="number" min="1" name="rounds" value="{int(config.get('num_rounds',10))}" required>
<label>Số client tối thiểu</label><input type="number" min="1" name="min_clients" value="{int(config.get('min_clients_connected',1))}" required>
<label>Cấu hình client (JSON)</label><textarea name="clients" required>{client_json}</textarea>
<button>Bắt đầu experiment mới</button></form></section>
<section class="card wide"><h2>Log hiện tại</h2><pre id="log">Đang tải…</pre></section>
<section class="card wide"><h2>Lịch sử</h2><table><thead><tr><th>Experiment</th><th>CSV</th><th>Model cuối</th></tr></thead><tbody>{rows}</tbody></table></section></div>
</main><script>async function logs(){{try{{document.getElementById('log').textContent=await (await fetch('/api/log')).text()}}catch(e){{}}}} logs();setInterval(logs,3000)</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def authenticated(self):
        expected = "Basic " + base64.b64encode(f"{ADMIN_USER}:{ADMIN_PASSWORD}".encode()).decode()
        if self.headers.get("Authorization") == expected:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="FL Dashboard"')
        self.end_headers()
        return False

    def respond(self, body, content_type="text/html; charset=utf-8", status=200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect_home(self):
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def do_GET(self):
        if not self.authenticated(): return
        route = urlparse(self.path).path
        if route == "/api/log":
            self.respond(manager.log(), "text/plain; charset=utf-8")
        elif route == "/api/status":
            self.respond(json.dumps(manager.status()), "application/json")
        else:
            self.respond(page())

    def do_POST(self):
        if not self.authenticated(): return
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        try:
            if self.path == "/start":
                config = {
                    "num_rounds": int(form["rounds"][0]),
                    "min_clients_connected": int(form["min_clients"][0]),
                    "clients": json.loads(form["clients"][0]),
                }
                if config["num_rounds"] < 1 or config["min_clients_connected"] < 1:
                    raise ValueError("Số round/client phải lớn hơn 0")
                manager.start(form.get("name", ["test"])[0], config)
                self.redirect_home()
            elif self.path == "/stop":
                manager.stop()
                self.redirect_home()
            else:
                self.respond("Not found", status=404)
        except Exception as exc:
            self.respond(page(f"Không thể thực hiện: {exc}"), status=400)

    def log_message(self, fmt, *args):
        print(f"[dashboard] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    host, port = UI_ADDRESS.rsplit(":", 1)
    if os.environ.get("FL_AUTOSTART", "true").lower() == "true":
        manager.start("startup", load_default_config())
    server = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"Dashboard: http://{host}:{port} | Flower: {FLOWER_ADDRESS}")
    try:
        server.serve_forever()
    finally:
        manager.stop()
