#!/usr/bin/env python3
"""
server.py — GestSense Bridge + Web Server
Team Ignite | Q-eHACK 2026

Does everything in ONE command:
  - Serves dashboard.html on HTTP port 8080
  - WebSocket on port 8765  (browser connects here)
  - Listens for UDP from QNX on port 5006:
      * ALERT   packets  → forward as {type:"alert", payload:...}
      * CANCEL  packets  → forward as {type:"cancel", payload:...}
  - Sends ACK back to QNX on port 5007 when dashboard ACK button clicked

Install:
    pip install websockets

Run:
    python server.py
"""

import asyncio, socket, json, threading, time, os, sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

# ─── CONFIG ───────────────────────────────────────────────────────
QNX_HOST         = "10.0.0.1"   # ← RPi4 QNX IP — change this
QNX_ACK_PORT     = 5007         # port to send ACK to on QNX

UDP_LISTEN_PORT  = 5006         # QNX sends alerts AND cancels here
WS_PORT          = 8765         # browser WebSocket connects here
HTTP_PORT        = 8080         # dashboard web server port

TELEGRAM_TOKEN   = ""
TELEGRAM_CHAT_ID = ""

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
# ──────────────────────────────────────────────────────────────────

clients       = set()
alert_history = []

# ─── HTTP SERVER (serves dashboard.html) ──────────────────────────
class DashboardHandler(SimpleHTTPRequestHandler):
    _html_cache = None

    @classmethod
    def load_html(cls):
        html_path = os.path.join(DASHBOARD_DIR, "dashboard.html")
        if os.path.exists(html_path):
            with open(html_path, "rb") as f:
                cls._html_cache = f.read()
            print(f"[HTTP] Loaded dashboard.html ({len(cls._html_cache)} bytes)")
        else:
            print(f"[HTTP] WARNING: dashboard.html not found at {html_path}")
            cls._html_cache = b"<h1>dashboard.html not found</h1>"

    def do_GET(self):
        content = self._html_cache or b"<h1>dashboard.html not found</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        pass

def run_http():
    DashboardHandler.load_html()
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), DashboardHandler)
    print(f"[HTTP] Serving dashboard on port {HTTP_PORT}")
    httpd.serve_forever()

# ─── TELEGRAM ─────────────────────────────────────────────────────
def send_telegram(a):
    if not TELEGRAM_TOKEN:
        return
    icons = {"AMBULANCE":"🚑","POLICE":"🚔","FIRE":"🔥","DISTRESS":"🆘"}
    icon  = icons.get(a.get("gesture",""), "🚨")
    gps   = "(est)"    if a.get("gps_synth") else "(GPS)"
    bme   = "(est)"    if a.get("bme_synth") else "(sensor)"
    text  = (
        f"{icon} *EMERGENCY — GestSense QNX*\n\n"
        f"*{a.get('gesture','')}* — {a.get('alert','')}\n"
        f"👤 {a.get('person','')}  🕐 {a.get('received','')}\n\n"
        f"📍 *GPS* {gps}\n"
        f"`{a.get('lat',0):.6f}, {a.get('lon',0):.6f}`\n"
        f"[Navigate](https://maps.google.com/maps?q={a.get('lat',0):.6f},{a.get('lon',0):.6f}&z=17&t=h)\n\n"
        f"🌡️ *Env* {bme}  "
        f"{a.get('temp',25):.1f}°C  {a.get('hum',40):.0f}%  {a.get('gas',100):.0f}kΩ"
    )
    try:
        import urllib.request
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID, "text": text,
            "parse_mode": "Markdown", "disable_web_page_preview": True
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8)
        print(f"[TELEGRAM] ✓ Sent")
    except Exception as e:
        print(f"[TELEGRAM] Error: {e}")

# ─── ACK → QNX ────────────────────────────────────────────────────
def send_ack_to_qnx(alert_id, gesture="", person=""):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        msg  = json.dumps({
            "type": "ACK",
            "alert_id": alert_id,
            "gesture": gesture,
            "person":  person,
            "time":    datetime.now().isoformat()
        }).encode()
        sock.sendto(msg, (QNX_HOST, QNX_ACK_PORT))
        sock.close()
        print(f"[ACK→QNX] Sent to {QNX_HOST}:{QNX_ACK_PORT}  gesture={gesture} person={person}")
    except Exception as e:
        print(f"[ACK→QNX] Error: {e}")

# ─── BROADCAST to all browser tabs ────────────────────────────────
async def broadcast(msg: str):
    dead = set()
    for ws in list(clients):
        try:    await ws.send(msg)
        except: dead.add(ws)
    clients.difference_update(dead)

# ─── WEBSOCKET HANDLER ────────────────────────────────────────────
async def ws_handler(websocket):
    clients.add(websocket)
    ip = websocket.remote_address[0]
    print(f"[WS] Browser connected: {ip}  (total: {len(clients)})")

    if alert_history:
        await websocket.send(json.dumps({
            "type": "history",
            "payload": alert_history[-20:]
        }))

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                if msg.get("type") == "ack":
                    aid     = msg.get("id")
                    gesture = msg.get("gesture", "")
                    # Find the original alert to pull person from it
                    person = ""
                    for a in alert_history:
                        if a.get("id") == aid:
                            person = a.get("person", "")
                            a["acked"] = True
                            break
                    print(f"[ACK] Dashboard acked  id={aid}  gesture={gesture}  person={person}")
                    send_ack_to_qnx(aid, gesture, person)
                    await broadcast(json.dumps({
                        "type": "ack_ok", "id": aid, "gesture": gesture
                    }))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        clients.discard(websocket)
        print(f"[WS] Browser disconnected  (remaining: {len(clients)})")

# ─── UDP LISTENER (from QNX) ──────────────────────────────────────
def udp_listener(loop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", UDP_LISTEN_PORT))
    print(f"[UDP] Listening for QNX messages on port {UDP_LISTEN_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            raw   = data.decode("utf-8", errors="ignore").strip()

            payload = json.loads(raw)
            msg_type = (payload.get("type") or "").upper()

            # ═══════════════════════════════════════════════════════
            # CANCEL packet from QNX (operator made fist gesture OR
            # dashboard ACK flow already pulsed buzzer → QNX broadcast)
            # ═══════════════════════════════════════════════════════
            if msg_type == "CANCEL":
                print(f"[UDP] {addr[0]} → CANCEL  gesture={payload.get('gesture','')}  person={payload.get('person','')}")

                # Mark all matching alerts in history as cancelled
                p_person  = (payload.get("person")  or "").strip()
                p_gesture = payload.get("gesture", "")
                for a in alert_history:
                    if (a.get("person") or "").strip() == p_person and not a.get("acked"):
                        a["acked"]     = True
                        a["cancelled"] = True

                # Forward to all browser tabs as type:"cancel"
                asyncio.run_coroutine_threadsafe(
                    broadcast(json.dumps({"type": "cancel", "payload": payload})),
                    loop
                )
                continue

            # ═══════════════════════════════════════════════════════
            # ALERT packet (primary or cascade)
            # ═══════════════════════════════════════════════════════
            cascade_tag = " [CASCADE]" if payload.get("cascade") else ""
            print(f"[UDP] {addr[0]} → ALERT{cascade_tag}  {payload.get('gesture','')}  {payload.get('person','')}")

            alert = payload
            alert["id"]       = int(time.time() * 1000000) + (hash(str(payload)) & 0xFFFF)
            alert["received"] = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            alert["acked"]    = False

            # Apply fallbacks — Shamshabad coordinates
            alert.setdefault("lat",       17.254973)
            alert.setdefault("lon",       78.308165)
            alert.setdefault("alt",       512.0)
            alert.setdefault("gps_synth", 1)
            alert.setdefault("temp",      36.5)
            alert.setdefault("hum",       32.0)
            alert.setdefault("gas",       180.0)
            alert.setdefault("bme_synth", 1)
            alert.setdefault("cascade",   False)
            alert.setdefault("cascade_of",  "")

            alert_history.append(alert)
            if len(alert_history) > 100:
                alert_history.pop(0)

            asyncio.run_coroutine_threadsafe(
                broadcast(json.dumps({"type": "alert", "payload": alert})),
                loop
            )

            # Telegram only for primary alerts (not cascade, not cancel)
            if not alert.get("cascade"):
                threading.Thread(
                    target=send_telegram, args=(alert,), daemon=True
                ).start()

        except json.JSONDecodeError:
            print(f"[UDP] Ignored bad JSON from {addr[0]}: {raw[:80]}")
        except Exception as e:
            print(f"[UDP] Error: {e}")

# ─── MAIN ─────────────────────────────────────────────────────────
async def main():
    import websockets

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "your-pc-ip"

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   GestSense Server — Team Ignite — Q-eHACK 2026 ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Dashboard  →  http://{local_ip}:{HTTP_PORT}           ║")
    print(f"║  WebSocket  →  ws://{local_ip}:{WS_PORT}           ║")
    print(f"║  UDP in     →  0.0.0.0:{UDP_LISTEN_PORT}  (from QNX)         ║")
    print(f"║  ACK out    →  {QNX_HOST}:{QNX_ACK_PORT}  (to QNX)         ║")
    if TELEGRAM_TOKEN:
        print(f"║  Telegram   →  ENABLED                          ║")
    else:
        print(f"║  Telegram   →  disabled (set TELEGRAM_TOKEN)    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Open on THIS PC   → http://localhost:{HTTP_PORT}")
    print(f"  Open on ANY PC    → http://{local_ip}:{HTTP_PORT}")
    print()

    threading.Thread(target=run_http, daemon=True).start()

    loop = asyncio.get_event_loop()
    threading.Thread(target=udp_listener, args=(loop,), daemon=True).start()

    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[server.py] Stopped.")
