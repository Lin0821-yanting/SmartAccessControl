#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai, Yanting Lin
# Tatung University — I4210 AI實務專題
"""scripts/analyze_decisions.py — 依情境標記統計決策 log。

吃 record_session.py 產出的統一 log（含 [DECISION] 逐幀行與 SCENARIO
START/END 標記），把每個情境切段，輸出 README「測試情境設計與統計方法」
那張結果表：每段的幀數、sim/live median、決策分布、開門/SPOOF 次數與判定。

用法
----
    # 分析最新一份 session log
    python scripts/analyze_decisions.py

    # 指定 log 檔
    python scripts/analyze_decisions.py logs/session_20260618_194524.log

    # 額外輸出 CSV
    python scripts/analyze_decisions.py --csv logs/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# [19:45:25.749] [DECISION] GRANT    identity=B    sim=0.712 live=0.401 frames=4/4
_DECISION_RE = re.compile(
    r"\[DECISION\]\s+(?P<decision>\S+)\s+identity=(?P<identity>\S+)\s+"
    r"sim=(?P<sim>[\d.]+)\s+live=(?P<live>[\d.]+)"
)
# ===== SCENARIO START [S9: 身分辨識正確性 (B) | 預期 identity=B] =====
_START_RE = re.compile(r"SCENARIO START \[(?P<id>[^:]+):\s*(?P<label>.+?)\s*\|\s*預期\s*(?P<expect>.+?)\]")
_END_RE = re.compile(r"SCENARIO END\s+\[(?P<id>[^:]+):")


@dataclass
class Segment:
    """單一情境分段累積的逐幀資料。"""

    scen_id: str
    label: str
    expect: str
    decisions: list[str] = field(default_factory=list)
    identities: list[str] = field(default_factory=list)
    sims: list[float] = field(default_factory=list)
    lives: list[float] = field(default_factory=list)

    def add(self, decision: str, identity: str, sim: float, live: float) -> None:
        """加入一幀。"""
        self.decisions.append(decision)
        self.identities.append(identity)
        self.sims.append(sim)
        self.lives.append(live)

    @property
    def n(self) -> int:
        """幀數。"""
        return len(self.decisions)


def _verdict(seg: Segment) -> str:
    """依預期與觀測決策給出 ✅ / ⚠️ 判定（best-effort）。"""
    if seg.n == 0:
        return "⚠️ 無資料"
    counts = Counter(seg.decisions)
    dominant = counts.most_common(1)[0][0]
    grants = counts.get("GRANT", 0)
    spoofs = counts.get("SPOOF", 0)
    exp = seg.expect.upper()

    if "GRANT" in exp:
        ok = grants > 0
    elif "不開門" in seg.expect:
        ok = grants == 0
    elif "非SPOOF" in seg.expect or "非 SPOOF" in seg.expect:
        ok = dominant != "SPOOF" and grants == 0
    elif "SPOOF" in exp:
        ok = dominant == "SPOOF" and grants == 0
    elif "UNKNOWN" in exp:
        ok = dominant == "UNKNOWN" and grants == 0
    elif "IGNORE" in exp:
        ok = dominant == "IGNORE"
    elif "IDENTITY=" in exp:
        want = exp.split("IDENTITY=", 1)[1].strip()
        got = Counter(self_id.upper() for self_id in seg.identities).most_common(1)[0][0]
        ok = grants > 0 and got == want
    else:
        ok = dominant == exp
    return f"{'✅' if ok else '⚠️'} 主決策={dominant}"


def _dist(seg: Segment) -> str:
    """決策分布字串（百分比）。"""
    counts = Counter(seg.decisions)
    parts = [f"{d}:{c / seg.n * 100:.0f}%" for d, c in counts.most_common()]
    return " ".join(parts)


def _med(values: list[float]) -> float:
    """中位數，空則 -1。"""
    return round(statistics.median(values), 3) if values else -1.0


def parse_log(path: Path) -> list[Segment]:
    """掃描 log，依 SCENARIO START/END 切出各情境分段。"""
    segments: list[Segment] = []
    current: Segment | None = None
    saw_marker = False

    for line in path.read_text(encoding="utf-8").splitlines():
        m_start = _START_RE.search(line)
        if m_start:
            saw_marker = True
            current = Segment(m_start["id"], m_start["label"], m_start["expect"])
            segments.append(current)
            continue
        if _END_RE.search(line):
            current = None
            continue
        m_dec = _DECISION_RE.search(line)
        if m_dec and current is not None:
            current.add(
                m_dec["decision"],
                m_dec["identity"],
                float(m_dec["sim"]),
                float(m_dec["live"]),
            )

    # 無任何標記 → 整份當一段
    if not saw_marker:
        whole = Segment("ALL", "整份 log（無情境標記）", "-")
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _DECISION_RE.search(line)
            if m:
                whole.add(m["decision"], m["identity"], float(m["sim"]), float(m["live"]))
        if whole.n:
            segments = [whole]
    return segments


def _latest_log(logdir: Path) -> Path | None:
    logs = sorted(logdir.glob("session_*.log"))
    return logs[-1] if logs else None


def main() -> None:
    """CLI 進入點。"""
    parser = argparse.ArgumentParser(description="依情境標記統計決策 log")
    parser.add_argument("log", nargs="?", type=Path, help="session log 路徑（預設用 logs/ 最新一份）")
    parser.add_argument("--logdir", type=Path, default=Path("logs"), help="自動尋找 log 的目錄")
    parser.add_argument("--csv", type=Path, default=None, help="另存 CSV 路徑")
    args = parser.parse_args()

    log_path = args.log or _latest_log(args.logdir)
    if log_path is None or not log_path.exists():
        print(f"找不到 log（{args.log or args.logdir}）。先用 record_session.py 錄製。")
        raise SystemExit(1)

    segments = parse_log(log_path)
    if not segments:
        print(f"{log_path} 內沒有可解析的 [DECISION] 行。")
        raise SystemExit(1)

    print(f"\n來源: {log_path}\n")
    header = ["情境", "幀數", "sim_med", "live_med", "決策分布", "開門", "SPOOF幀", "判定"]
    rows: list[list[str]] = []
    for seg in segments:
        counts = Counter(seg.decisions)
        rows.append(
            [
                f"{seg.scen_id} {seg.label}",
                str(seg.n),
                f"{_med(seg.sims):.3f}",
                f"{_med(seg.lives):.3f}",
                _dist(seg),
                str(counts.get("GRANT", 0)),
                str(counts.get("SPOOF", 0)),
                _verdict(seg),
            ]
        )

    # Markdown 表
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()

    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"CSV 已輸出: {args.csv}")


if __name__ == "__main__":
    main()
