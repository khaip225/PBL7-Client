"""
PBL7 Client Monitoring Dashboard
Streamlit-based real-time monitoring for federated learning clients.

Usage:
    streamlit run dashboard/app.py
    streamlit run dashboard/app.py -- --state-file ./client_state.json --server-url http://127.0.0.1:8000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import streamlit as st
import requests

# ---- Page config ----
st.set_page_config(
    page_title="PBL7 Client Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state-file", default="./client_state.json")
    p.add_argument("--server-url", default="http://127.0.0.1:8000")
    p.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    return p.parse_args()


args = parse_args()
STATE_FILE = args.state_file
SERVER_URL = args.server_url.rstrip("/")
REFRESH = args.refresh


# ---- Helpers ----
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def fetch_server_health() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---- UI ----
st.title("🏥 PBL7 Client Monitoring")
st.caption(f"State file: `{STATE_FILE}` | Server: `{SERVER_URL}` | Refresh: {REFRESH}s")

# Auto-refresh
placeholder = st.empty()

while True:
    state = load_state()
    server_ok = fetch_server_health()

    with placeholder.container():
        # ==================================================================
        # Status Panel
        # ==================================================================
        st.header("Status")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            server_label = "✅ Connected" if state.get("connected_to_server") else "❌ Disconnected"
            st.metric("FastAPI Server", server_label)
        with c2:
            flower_label = "✅ Connected" if state.get("connected_to_flower") else "❌ Not Connected"
            st.metric("Flower Server", flower_label)
        with c3:
            cid = state.get("client_id") or "N/A"
            st.metric("Client ID", cid if len(str(cid)) < 12 else str(cid)[:12] + "...")
        with c4:
            st.metric("Modality", state.get("modality", "N/A").upper())
        with c5:
            st.metric("Status", state.get("status", "offline").upper())

        st.divider()

        # ==================================================================
        # System Metrics + Training Status
        # ==================================================================
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("System Metrics")
            sys_metrics = state.get("system", {})
            if sys_metrics:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    cpu = sys_metrics.get("cpu_percent", 0)
                    st.metric("CPU", f"{cpu}%")
                    st.progress(cpu / 100, text=f"CPU {cpu:.1f}%")
                with mc2:
                    ram = sys_metrics.get("ram_percent", 0)
                    st.metric("RAM", f"{ram}%")
                    st.progress(ram / 100, text=f"RAM {ram:.1f}%")
                with mc3:
                    gpu = sys_metrics.get("gpu_percent", 0)
                    gpu_temp = sys_metrics.get("gpu_temp")
                    temp_str = f" | {gpu_temp}°C" if gpu_temp else ""
                    st.metric("GPU", f"{gpu}%{temp_str}")
                    if gpu > 0:
                        st.progress(gpu / 100, text=f"GPU {gpu:.1f}%")
                    else:
                        st.caption("No GPU detected")

                # Disk
                disk = sys_metrics.get("disk_percent", 0)
                if disk:
                    st.metric("Disk", f"{disk}%")
                    st.progress(disk / 100, text=f"Disk {disk:.1f}%")
            else:
                st.info("No system metrics yet — waiting for heartbeat data")

        with col_right:
            st.subheader("Training Status")
            current_round = state.get("current_round", 0)
            total_rounds = state.get("total_rounds", 0)

            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric("Current Round", f"{current_round}/{total_rounds}")
            with tc2:
                loss = state.get("loss")
                st.metric("Loss", f"{loss:.4f}" if loss is not None else "N/A")
            with tc3:
                acc = state.get("accuracy")
                st.metric("Accuracy", f"{acc:.2%}" if acc is not None else "N/A")

            if total_rounds > 0:
                pct = min(current_round / total_rounds, 1.0)
                st.progress(pct, text=f"Progress: {pct:.0%}")

        st.divider()

        # ==================================================================
        # Network Status
        # ==================================================================
        st.subheader("Network")
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            hb = state.get("last_heartbeat")
            if hb:
                try:
                    hb_dt = datetime.fromisoformat(hb)
                    ago = (datetime.now(timezone.utc) - hb_dt).total_seconds()
                    st.metric("Last Heartbeat", f"{ago:.0f}s ago")
                except Exception:
                    st.metric("Last Heartbeat", hb)
            else:
                st.metric("Last Heartbeat", "Never")

        with nc2:
            lat = state.get("latency_ms", 0)
            st.metric("Latency", f"{lat:.1f} ms")

        with nc3:
            st.metric("Server Health", "OK" if server_ok else "UNREACHABLE")

        st.divider()

        # ==================================================================
        # Logs Panel
        # ==================================================================
        st.subheader("Logs")
        logs = state.get("logs", [])
        if logs:
            # Show last 50 logs in a scrollable container
            log_text = ""
            for entry in reversed(logs[-50:]):
                ts = entry.get("timestamp", "")[:19].replace("T", " ")
                level = entry.get("level", "info").upper()
                msg = entry.get("message", "")
                emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🔥"}.get(level, "")
                log_text += f"`{ts}` {emoji} **[{level}]** {msg}\n\n"
            st.markdown(log_text)
        else:
            st.info("No logs yet")

    time.sleep(REFRESH)
    st.rerun()
