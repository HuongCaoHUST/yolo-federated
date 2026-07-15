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
        self.current_config = None
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
            self.current_config = config
            return run_id

    def config(self):
        with self.lock:
            if self.current_config is not None:
                return self.current_config
            return load_default_config()

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

    def progress(self):
        """Build a friendly status snapshot from Flower's current log."""
        with self.lock:
            status = self.status()
            config = self.config()
            total_rounds = int(config.get("num_rounds", 0))
            minimum_clients = int(config.get("min_clients_connected", 0))
            text = self.log(lines=1500) if self.current_run else ""

            rounds = [int(value) for value in re.findall(
                r"(?:fit_round|aggregate_fit|\[ROUND)\s*[:\[]?\s*(\d+)", text, re.IGNORECASE
            )]
            current_round = max(rounds, default=0)
            sampled = [int(value) for value in re.findall(r"sampled\s+(\d+)\s+client", text, re.IGNORECASE)]
            connected_clients = sampled[-1] if sampled else 0
            if "Received initial parameters from one random client" in text:
                connected_clients = max(connected_clients, 1)

            markers = [
                ("Đang chờ client kết nối", max(text.rfind("Requesting initial parameters"), text.rfind("Waiting for"))),
                ("Đang gửi model đến client", max(text.rfind("configure_fit"), text.rfind("strategy sampled"))),
                ("Client đang huấn luyện cục bộ", max(text.rfind("fit_round"), text.rfind("fit progress"))),
                ("Đang tổng hợp trọng số", max(text.rfind("aggregate_fit"), text.rfind("Fusion globale"))),
                ("Đã hoàn thành huấn luyện", max(text.rfind("Apprentissage Fédéré terminé"), text.rfind("FL finished"))),
            ]
            phase, marker_position = max(markers, key=lambda item: item[1])
            if marker_position < 0:
                phase = "Đang khởi động server" if status["running"] else "Chưa bắt đầu"
            if not status["running"] and current_round >= total_rounds > 0:
                phase = "Đã hoàn thành huấn luyện"
            elif not status["running"] and status["run"]:
                phase = "Đã dừng"

            percent = round(min(current_round / total_rounds * 100, 100)) if total_rounds else 0
            if phase == "Đã hoàn thành huấn luyện":
                percent = 100

            elapsed = 0
            if self.current_run:
                run_dir = RUNS_DIR / self.current_run
                if run_dir.exists():
                    elapsed = max(0, int(time.time() - run_dir.stat().st_mtime))
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            validation = {"running": False, "percent": 0, "message": "Chưa chạy validation"}
            if self.current_run:
                validation_path = RUNS_DIR / self.current_run / "validation_status.json"
                if validation_path.exists():
                    try:
                        validation = json.loads(validation_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        pass
            if validation.get("running"):
                phase = f"Đang validation global model — Round {validation.get('round', current_round)}"

            return {
                **status,
                "phase": phase,
                "current_round": current_round,
                "total_rounds": total_rounds,
                "percent": percent,
                "connected_clients": connected_clients,
                "minimum_clients": minimum_clients,
                "elapsed": elapsed_text,
                "validation": validation,
            }

    def global_metrics(self):
        with self.lock:
            if not self.current_run:
                return []
            path = RUNS_DIR / self.current_run / "global_metrics.json"
            if not path.exists():
                return []
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []

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
    config = manager.config()
    status = manager.status()
    runs = manager.runs()
    state = "ĐANG CHẠY" if status["running"] else "ĐÃ DỪNG"
    state_class = "running" if status["running"] else "stopped"
    rows = "".join(
        f"<tr><td>{html.escape(run['name'])}</td><td>{'✓' if run['report'] else '–'}</td>"
        f"<td>{'✓' if run['model'] else '–'}</td></tr>" for run in runs
    ) or "<tr><td colspan='3'>Chưa có experiment</td></tr>"
    client_json = html.escape(json.dumps(config.get("clients", {}), ensure_ascii=False, indent=2))
    validation = config.get("global_validation", {})
    validation_checked = "checked" if validation.get("enabled", False) else ""
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO Federated Learning</title>
<style>
:root{{--card:rgba(255,255,255,.8);--line:rgba(190,24,93,.13);--text:#351326;--muted:#846477;--pink:#ec4899;--pink-dark:#be185d;--purple:#a855f7;--danger:#e11d48}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;color:var(--text);font:15px Inter,"SF Pro Display","Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#fff7fb;background-image:radial-gradient(circle at 8% 5%,rgba(251,207,232,.95),transparent 30%),radial-gradient(circle at 92% 12%,rgba(233,213,255,.8),transparent 28%),linear-gradient(145deg,#fffafd 10%,#fdf2f8 52%,#faf5ff);background-attachment:fixed}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(255,255,255,.35) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.35) 1px,transparent 1px);background-size:38px 38px;mask-image:linear-gradient(to bottom,#000,transparent 72%)}}
.wrap{{position:relative;max-width:1120px;margin:auto;padding:46px 20px 64px}}
h1{{margin:0 0 8px;font-size:clamp(30px,4vw,43px);line-height:1.08;letter-spacing:-1.4px;font-weight:850;background:linear-gradient(110deg,#9d174d,#ec4899 48%,#9333ea);-webkit-background-clip:text;background-clip:text;color:transparent}}
h2{{margin:0 0 18px;font-size:17px;letter-spacing:-.3px}} .sub{{color:var(--muted);margin-bottom:30px;font-size:16px}}
.grid{{display:grid;grid-template-columns:minmax(0,.82fr) minmax(0,1.18fr);gap:20px}}
.card{{position:relative;overflow:hidden;background:var(--card);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid rgba(255,255,255,.94);border-radius:22px;padding:24px;box-shadow:0 18px 50px rgba(157,23,77,.09),0 3px 12px rgba(91,33,182,.05);transition:transform .25s ease,box-shadow .25s ease}}
.card:hover{{transform:translateY(-2px);box-shadow:0 24px 58px rgba(157,23,77,.13)}} .card:after{{content:"";position:absolute;width:130px;height:130px;right:-72px;top:-78px;border-radius:50%;background:linear-gradient(135deg,rgba(236,72,153,.15),rgba(168,85,247,.12));pointer-events:none}}
.wide{{grid-column:1/-1}} label{{display:block;color:#745467;margin:14px 0 7px;font-size:13px;font-weight:700;letter-spacing:.15px}}
input,textarea{{width:100%;background:rgba(255,255,255,.86);color:var(--text);border:1px solid rgba(190,24,93,.16);border-radius:12px;padding:12px 13px;box-shadow:inset 0 1px 2px rgba(80,20,50,.03);transition:border-color .2s,box-shadow .2s,background .2s}}
input:focus,textarea:focus{{outline:none;background:#fff;border-color:var(--pink);box-shadow:0 0 0 4px rgba(236,72,153,.12)}} textarea{{min-height:235px;resize:vertical;font:13px/1.55 "SFMono-Regular",Consolas,monospace}}
button{{border:0;border-radius:12px;padding:12px 18px;color:#fff;background:linear-gradient(120deg,var(--pink-dark),var(--pink) 58%,var(--purple));cursor:pointer;font:700 14px Inter,"Segoe UI",sans-serif;margin:14px 8px 0 0;box-shadow:0 9px 22px rgba(219,39,119,.25);transition:transform .18s,box-shadow .18s,filter .18s}}
button:hover{{transform:translateY(-1px);box-shadow:0 13px 28px rgba(219,39,119,.32);filter:saturate(1.08)}} button:active{{transform:none}} button.danger{{background:linear-gradient(120deg,#be123c,#fb7185);box-shadow:0 9px 22px rgba(225,29,72,.2)}}
.badge{{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.4px}} .badge:before{{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 0 4px rgba(255,255,255,.65)}} .running{{background:#dcfce7;color:#087c56}} .stopped{{background:#ffe4e8;color:#be123c}}
pre{{background:#2b1322;color:#fce7f3;border:1px solid rgba(236,72,153,.17);border-radius:15px;padding:18px;max-height:430px;overflow:auto;white-space:pre-wrap;font:12.5px/1.65 "SFMono-Regular",Consolas,monospace;box-shadow:inset 0 1px 12px rgba(0,0,0,.08)}}
table{{width:100%;border-collapse:separate;border-spacing:0}} th{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.7px}} td,th{{padding:12px 10px;border-bottom:1px solid var(--line);text-align:left}} tbody tr{{transition:background .18s}} tbody tr:hover{{background:rgba(252,231,243,.58)}} tbody tr:last-child td{{border-bottom:0}}
.msg{{color:#831843;background:linear-gradient(110deg,#fce7f3,#f3e8ff);border:1px solid rgba(236,72,153,.15);padding:12px 15px;border-radius:13px;margin-bottom:18px;font-weight:650}}
.phase{{margin:19px 0 6px;font-size:18px;font-weight:800;letter-spacing:-.35px}} .experiment{{color:var(--muted);font-size:13px;overflow-wrap:anywhere}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:19px 0}} .stat{{background:rgba(253,242,248,.75);border:1px solid var(--line);border-radius:14px;padding:12px 9px;text-align:center}} .stat-value{{display:block;font-size:19px;font-weight:850;color:#9d174d}} .stat-label{{display:block;margin-top:3px;color:var(--muted);font-size:10px;font-weight:750;text-transform:uppercase;letter-spacing:.45px}}
.progress-head{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;font-weight:700;margin-bottom:7px}} .progress-track{{height:10px;overflow:hidden;background:#fce7f3;border-radius:999px;box-shadow:inset 0 1px 2px rgba(131,24,67,.08)}} .progress-fill{{height:100%;width:0;border-radius:inherit;background:linear-gradient(90deg,#db2777,#ec4899,#a855f7);box-shadow:0 0 12px rgba(236,72,153,.4);transition:width .5s ease}}
.validation-box{{margin-top:16px;padding:13px;background:rgba(250,245,255,.8);border:1px solid rgba(168,85,247,.14);border-radius:14px}} .validation-box[hidden]{{display:none}} .validation-title{{display:flex;justify-content:space-between;margin-bottom:8px;font-size:12px;font-weight:800;color:#7e22ce}}
.check-row{{display:flex;align-items:center;gap:9px;margin-top:16px;color:#745467;font-weight:700}} .check-row input{{width:17px;height:17px;accent-color:var(--pink);box-shadow:none}} .form-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:15px}} .chart-box{{background:rgba(255,255,255,.66);border:1px solid var(--line);border-radius:16px;padding:14px}} .chart-title{{font-size:13px;font-weight:800;color:#831843;margin-bottom:8px}} canvas{{display:block;width:100%;height:260px}} .chart-empty{{color:var(--muted);font-size:13px;text-align:center;padding:30px}}
details{{border:1px solid var(--line);border-radius:15px;background:rgba(255,255,255,.55);overflow:hidden}} summary{{cursor:pointer;list-style:none;padding:14px 17px;color:#831843;font-weight:750;user-select:none}} summary::-webkit-details-marker{{display:none}} summary:after{{content:"＋";float:right;font-size:19px;line-height:15px}} details[open] summary:after{{content:"−"}} details pre{{margin:0;border:0;border-radius:0}}
@media(max-width:760px){{.wrap{{padding-top:30px}}.grid,.chart-grid{{grid-template-columns:1fr}}.form-row{{grid-template-columns:1fr}}.card{{padding:19px;border-radius:18px}}}}
</style></head><body><main class="wrap">
<h1>YOLO Federated Learning</h1><div class="sub">Quản lý các lượt huấn luyện</div>
{f'<div class="msg">{html.escape(message)}</div>' if message else ''}
<div class="grid"><section class="card"><h2>Trạng thái huấn luyện</h2><span id="stateBadge" class="badge {state_class}">{state}</span>
<div id="phase" class="phase">Đang cập nhật…</div><div class="experiment">Experiment: <b id="runName">{html.escape(status['run'] or '–')}</b></div>
<div class="stats"><div class="stat"><span id="roundValue" class="stat-value">0/0</span><span class="stat-label">Round</span></div><div class="stat"><span id="clientValue" class="stat-value">0/0</span><span class="stat-label">Client</span></div><div class="stat"><span id="elapsedValue" class="stat-value">00:00:00</span><span class="stat-label">Thời gian</span></div></div>
<div class="progress-head"><span>Tiến độ tổng thể</span><span id="percentValue">0%</span></div><div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
<div id="validationBox" class="validation-box" hidden><div class="validation-title"><span id="validationMessage">Global validation</span><span id="validationPercent">0%</span></div><div class="progress-track"><div id="validationFill" class="progress-fill"></div></div></div>
<form method="post" action="/stop"><button class="danger">Dừng server</button></form></section>
<section class="card"><h2>Tạo lượt chạy mới</h2><form method="post" action="/start">
<label>Tên test</label><input name="name" value="test" required>
<label>Số round</label><input type="number" min="1" name="rounds" value="{int(config.get('num_rounds',10))}" required>
<label>Số client tối thiểu</label><input type="number" min="1" name="min_clients" value="{int(config.get('min_clients_connected',1))}" required>
<label>Cấu hình client (JSON)</label><textarea name="clients" required>{client_json}</textarea>
<label class="check-row"><input type="checkbox" name="validation_enabled" {validation_checked}> Validation global model sau mỗi round</label>
<label>Global validation data.yaml trong container</label><input name="validation_data" value="{html.escape(str(validation.get('data', '/app/coco8.yaml')))}">
<div class="form-row"><div><label>Batch size</label><input type="number" min="1" name="validation_batch" value="{int(validation.get('batch_size', 8))}"></div><div><label>Image size</label><input type="number" min="32" name="validation_imgsz" value="{int(validation.get('imgsz', 320))}"></div><div><label>Workers</label><input type="number" min="0" name="validation_workers" value="{int(validation.get('workers', 2))}"></div></div>
<button>Bắt đầu experiment mới</button></form></section>
<section class="card wide"><h2>Global validation theo round</h2><div id="chartEmpty" class="chart-empty">Biểu đồ sẽ xuất hiện sau khi validation round đầu tiên hoàn tất.</div><div id="chartGrid" class="chart-grid" hidden><div class="chart-box"><div class="chart-title">Precision · Recall · mAP</div><canvas id="qualityChart" width="900" height="300"></canvas></div><div class="chart-box"><div class="chart-title">Validation loss</div><canvas id="lossChart" width="900" height="300"></canvas></div></div></section>
<section class="card wide"><h2>Chi tiết kỹ thuật</h2><details id="logDetails"><summary>Xem log Flower server</summary><pre id="log">Mở mục này để tải log…</pre></details></section>
<section class="card wide"><h2>Lịch sử</h2><table><thead><tr><th>Experiment</th><th>CSV</th><th>Model cuối</th></tr></thead><tbody>{rows}</tbody></table></section></div>
</main><script>
async function refreshStatus(){{try{{const s=await (await fetch('/api/progress')).json();const badge=document.getElementById('stateBadge');badge.textContent=s.running?'ĐANG CHẠY':'ĐÃ DỪNG';badge.className='badge '+(s.running?'running':'stopped');document.getElementById('phase').textContent=s.phase;document.getElementById('runName').textContent=s.run||'–';document.getElementById('roundValue').textContent=s.current_round+'/'+s.total_rounds;document.getElementById('clientValue').textContent=s.connected_clients+'/'+s.minimum_clients;document.getElementById('elapsedValue').textContent=s.elapsed;document.getElementById('percentValue').textContent=s.percent+'%';document.getElementById('progressFill').style.width=s.percent+'%';const v=s.validation||{{}};const vb=document.getElementById('validationBox');vb.hidden=!(v.running||v.percent||v.error);document.getElementById('validationMessage').textContent=v.error?'Validation lỗi: '+v.error:(v.message||'Global validation');document.getElementById('validationPercent').textContent=(v.percent||0)+'%';document.getElementById('validationFill').style.width=(v.percent||0)+'%'}}catch(e){{}}}}
function drawChart(id,data,series){{const c=document.getElementById(id),x=c.getContext('2d'),w=c.width,h=c.height,p={{l:50,r:18,t:30,b:38}};x.clearRect(0,0,w,h);const values=data.flatMap(d=>series.map(s=>Number(d[s.key]||0))),max=Math.max(...values,series.some(s=>s.fixed)?1:0.001),min=0;x.strokeStyle='#ecd7e3';x.lineWidth=1;x.font='12px Inter, sans-serif';x.fillStyle='#8b6a7e';for(let i=0;i<=4;i++){{const y=p.t+(h-p.t-p.b)*i/4;x.beginPath();x.moveTo(p.l,y);x.lineTo(w-p.r,y);x.stroke();x.fillText((max*(1-i/4)).toFixed(max<=1?2:3),5,y+4)}}const px=i=>p.l+(w-p.l-p.r)*(data.length===1?.5:i/(data.length-1));const py=v=>h-p.b-(Number(v)-min)/(max-min||1)*(h-p.t-p.b);data.forEach((d,i)=>x.fillText('R'+d.round,px(i)-8,h-13));series.forEach((s,si)=>{{x.strokeStyle=s.color;x.lineWidth=3;x.beginPath();data.forEach((d,i)=>{{const xx=px(i),yy=py(d[s.key]||0);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}});x.stroke();data.forEach((d,i)=>{{x.fillStyle=s.color;x.beginPath();x.arc(px(i),py(d[s.key]||0),4,0,Math.PI*2);x.fill()}});x.fillStyle=s.color;x.fillRect(p.l+si*145,8,12,3);x.fillText(s.label,p.l+17+si*145,13)}})}}
async function refreshCharts(){{try{{const data=await (await fetch('/api/metrics')).json();document.getElementById('chartEmpty').hidden=data.length>0;document.getElementById('chartGrid').hidden=!data.length;if(!data.length)return;drawChart('qualityChart',data,[{{key:'precision',label:'Precision',color:'#ec4899',fixed:true}},{{key:'recall',label:'Recall',color:'#a855f7',fixed:true}},{{key:'map50',label:'mAP50',color:'#f97316',fixed:true}},{{key:'map50_95',label:'mAP50-95',color:'#0ea5e9',fixed:true}}]);drawChart('lossChart',data,[{{key:'box_loss',label:'Box loss',color:'#ec4899'}},{{key:'obj_loss',label:'Obj loss',color:'#a855f7'}},{{key:'cls_loss',label:'Cls loss',color:'#f97316'}},{{key:'total_loss',label:'Total loss',color:'#0f9f6e'}}])}}catch(e){{}}}}
async function logs(){{if(!document.getElementById('logDetails').open)return;try{{document.getElementById('log').textContent=await (await fetch('/api/log')).text()}}catch(e){{}}}}
refreshStatus();refreshCharts();setInterval(refreshStatus,2000);setInterval(refreshCharts,5000);document.getElementById('logDetails').addEventListener('toggle',logs);setInterval(logs,3000)
</script></body></html>"""


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
        elif route == "/api/progress":
            self.respond(json.dumps(manager.progress()), "application/json")
        elif route == "/api/metrics":
            self.respond(json.dumps(manager.global_metrics()), "application/json")
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
                    "global_validation": {
                        "enabled": "validation_enabled" in form,
                        "data": form.get("validation_data", ["/app/coco8.yaml"])[0],
                        "batch_size": int(form.get("validation_batch", [8])[0]),
                        "imgsz": int(form.get("validation_imgsz", [320])[0]),
                        "workers": int(form.get("validation_workers", [2])[0]),
                    },
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
