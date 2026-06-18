#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai, Yanting Lin
# Tatung University — I4210 AI實務專題
"""scripts/record_session.py — 測試情境錄製與標記工具。

包住主程式（main.py）執行，把主程式每一行輸出加上時間戳寫進統一 log，
並提供「情境起訖標記 + 開始前倒數緩衝」，讓你有時間就定位/拿照片，避免
誤操作。事後可依標記把 log 切成每個情境分段做統計。

用法
----
    # 包住主程式錄製（建議）
    python scripts/record_session.py -- pdm run python src/pipeline/main.py --no-display

    # 只標記、不啟動主程式（主程式在別處/容器內跑時，用 wall-clock 對齊）
    python scripts/record_session.py --no-launch

    # 調整倒數秒數
    python scripts/record_session.py --buffer 8 -- pdm run python src/pipeline/main.py --no-display

互動指令（執行中於本視窗輸入）
------------------------------
    1..N    開始某情境（會先倒數 --buffer 秒）→ 寫入 START 標記
    Enter   結束目前情境 → 寫入 END 標記
    l       列出情境清單
    q       結束並關閉主程式

產出
----
    logs/session_<時間>.log           主程式輸出（每行帶時間戳）+ 情境標記
    logs/session_<時間>.markers.jsonl 每個情境的結構化起訖（給分析腳本用）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 預設測試情境（對應 README「測試情境設計與統計方法」）
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict[str, str]] = {
    "1": {"id": "S1", "label": "註冊者真人 (B)", "expect": "GRANT"},
    "2": {"id": "S2", "label": "註冊者照片 (B)", "expect": "SPOOF"},
    "3": {"id": "S3", "label": "註冊者螢幕 (B)", "expect": "SPOOF"},
    "4": {"id": "S4", "label": "陌生人真人 (C)", "expect": "UNKNOWN"},
    "5": {"id": "S5", "label": "陌生人照片 (C)", "expect": "SPOOF"},
    "6": {"id": "S6", "label": "無人", "expect": "IGNORE"},
    "7": {"id": "S7", "label": "一閃而過 (B)", "expect": "不開門"},
    "8": {"id": "S8", "label": "距離太遠 (B)", "expect": "IGNORE"},
    "9": {"id": "S9", "label": "身分辨識正確性 (B)", "expect": "identity=B"},
    "10": {"id": "S10", "label": "對照組:真人變裝 (B)", "expect": "非SPOOF"},
}

DEFAULT_BUFFER_S = 5
_DEFAULT_CMD = ["pdm", "run", "python", "src/pipeline/main.py", "--no-display"]


def _now() -> str:
    """毫秒級時間戳字串。"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class SessionRecorder:
    """管理統一 log、情境標記，以及（可選）被包住的主程式子行程。"""

    def __init__(self, outdir: Path, buffer_s: int) -> None:
        """建立 log/markers 檔案與寫入鎖。"""
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = outdir / f"session_{stamp}.log"
        self.markers_path = outdir / f"session_{stamp}.markers.jsonl"
        self.buffer_s = buffer_s
        self._log = self.log_path.open("w", encoding="utf-8")
        self._markers = self.markers_path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._active: dict[str, str] | None = None
        self._active_start: float = 0.0

    # ── 寫入 ────────────────────────────────────────────────────────────
    def write_line(self, text: str, *, echo: bool = False) -> None:
        """寫一行帶時間戳的內容到統一 log（執行緒安全）。"""
        line = f"[{_now()}] {text}"
        with self._lock:
            self._log.write(line + "\n")
            self._log.flush()
        if echo:
            print(line, flush=True)

    def _write_marker(self, kind: str, scen: dict[str, str], extra: dict) -> None:
        rec = {
            "kind": kind,
            "scenario": scen["id"],
            "label": scen["label"],
            "expect": scen["expect"],
            "ts": time.time(),
            "iso": datetime.now().isoformat(timespec="milliseconds"),
            **extra,
        }
        with self._lock:
            self._markers.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._markers.flush()

    # ── 情境起訖 ────────────────────────────────────────────────────────
    def start_scenario(self, key: str) -> None:
        """倒數緩衝後標記某情境開始。"""
        scen = SCENARIOS.get(key)
        if scen is None:
            print(f"  未知情境 '{key}'，輸入 l 看清單。", flush=True)
            return
        if self._active is not None:
            self.end_scenario()  # 自動收尾前一段

        print(f"\n>>> 準備情境 {scen['id']} {scen['label']}（預期 {scen['expect']}）", flush=True)
        for remaining in range(self.buffer_s, 0, -1):
            print(f"    就定位… {remaining}", end="\r", flush=True)
            time.sleep(1)
        print("    開始錄製！          ", flush=True)

        self._active = scen
        self._active_start = time.time()
        banner = f"===== SCENARIO START [{scen['id']}: {scen['label']} | 預期 {scen['expect']}] ====="
        self.write_line(banner)
        self._write_marker("start", scen, {})

    def end_scenario(self) -> None:
        """標記目前情境結束。"""
        if self._active is None:
            print("  目前沒有進行中的情境。", flush=True)
            return
        scen = self._active
        dur = round(time.time() - self._active_start, 2)
        banner = f"===== SCENARIO END   [{scen['id']}: {scen['label']}] 歷時 {dur}s ====="
        self.write_line(banner)
        self._write_marker("end", scen, {"duration_s": dur})
        print(f"<<< 結束 {scen['id']}（{dur}s）\n", flush=True)
        self._active = None

    def close(self) -> None:
        """收尾並關閉檔案。"""
        if self._active is not None:
            self.end_scenario()
        with self._lock:
            self._log.close()
            self._markers.close()


def _pump_child_output(proc: subprocess.Popen, rec: SessionRecorder) -> None:
    """讀取子行程 stdout，逐行加時間戳寫入 log 並回顯。"""
    assert proc.stdout is not None
    for raw in proc.stdout:
        rec.write_line(raw.rstrip("\n"), echo=True)


def _print_menu() -> None:
    print("\n情境清單（輸入編號開始；Enter 結束目前段；l 清單；q 離開）:", flush=True)
    for key, scen in SCENARIOS.items():
        print(f"  {key:>2}  {scen['id']:<4} {scen['label']:<18} → 預期 {scen['expect']}", flush=True)
    print(flush=True)


def main() -> None:
    """CLI 進入點。"""
    parser = argparse.ArgumentParser(description="測試情境錄製與標記工具")
    parser.add_argument(
        "--buffer", type=int, default=DEFAULT_BUFFER_S, help="每段開始前的倒數緩衝秒數"
    )
    parser.add_argument("--outdir", type=Path, default=Path("logs"), help="輸出目錄")
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="不啟動主程式，只做標記（主程式在別處/容器內跑時用）",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="-- 後接要包住的主程式指令（預設 pdm run python src/pipeline/main.py --no-display）",
    )
    args = parser.parse_args()

    # argparse.REMAINDER 會把開頭的 '--' 一起收進來，去掉它。
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd and not args.no_launch:
        cmd = _DEFAULT_CMD

    rec = SessionRecorder(args.outdir, args.buffer)
    print(f"[record] log     → {rec.log_path}")
    print(f"[record] markers → {rec.markers_path}")
    print(f"[record] 倒數緩衝 = {rec.buffer_s}s")

    proc: subprocess.Popen | None = None
    if not args.no_launch:
        print(f"[record] 啟動主程式: {' '.join(cmd)}\n")
        proc = subprocess.Popen(  # noqa: S603 — 指令由使用者於本機提供
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=_pump_child_output, args=(proc, rec), daemon=True).start()
    else:
        print("[record] --no-launch：只做標記，請自行在別處啟動主程式。\n")

    _print_menu()

    try:
        while True:
            try:
                cmd_in = input().strip()
            except EOFError:
                break
            if cmd_in == "":
                rec.end_scenario()
            elif cmd_in.lower() == "q":
                break
            elif cmd_in.lower() == "l":
                _print_menu()
            elif cmd_in in SCENARIOS:
                rec.start_scenario(cmd_in)
            else:
                print(f"  無效輸入 '{cmd_in}'（l 看清單）。", flush=True)
            # 子行程提早結束就跳出
            if proc is not None and proc.poll() is not None:
                print("[record] 主程式已結束。", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[record] 中斷。", flush=True)
    finally:
        rec.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"[record] 完成。log: {rec.log_path}")


if __name__ == "__main__":
    main()
