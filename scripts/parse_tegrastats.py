#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai, Yanting Lin
# Tatung University — I4210 AI實務專題
"""scripts/parse_tegrastats.py — tegrastats.log → utilization.csv (+ summary).

Parses a tegrastats capture into the CSV schema required by the capstone
submission (§B.8) and prints mean / p50 / p95 / max per column for the report
§6.3 resource & power table.

Capture (on the Jetson, ≥ 60 s under sustained inference load):
    sudo tegrastats --interval 1000 --logfile tegrastats.log

Parse:
    python scripts/parse_tegrastats.py tegrastats.log --out utilization.csv

Paths can be overridden by CLI args or env vars so a grader can re-run this
against the unzipped test_artifacts folder:
    TEGRASTATS_LOG=/path/tegrastats.log UTILIZATION_CSV=/path/utilization.csv \
        python scripts/parse_tegrastats.py

CSV columns (§B.8):
    t, cpu_avg_pct, gpu_pct, ram_used_mb,
    vdd_in_mw, vdd_cpu_mw, vdd_gpu_mw, vdd_soc_mw,
    gpu_temp_c, cpu_temp_c

Note on Orin Nano power rails: JetPack 6 reports a combined VDD_CPU_GPU_CV rail
(no separate VDD_CPU / VDD_GPU). That combined value is written to vdd_gpu_mw and
vdd_cpu_mw is left blank; VDD_IN and VDD_SOC are reported as-is.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
from pathlib import Path

_COLUMNS = [
    "t",
    "cpu_avg_pct",
    "gpu_pct",
    "ram_used_mb",
    "vdd_in_mw",
    "vdd_cpu_mw",
    "vdd_gpu_mw",
    "vdd_soc_mw",
    "gpu_temp_c",
    "cpu_temp_c",
]

# tegrastats fragments
_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
_CPU_RE = re.compile(r"CPU \[([^\]]*)\]")
_CPU_CORE_RE = re.compile(r"(\d+)%@")
_GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
_RAIL_RE = re.compile(r"(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+|POM_[A-Z0-9_]+) (\d+)mW")
_TEMP_RE = re.compile(r"\b([a-zA-Z0-9_]+)@(-?[\d.]+)C")
# Optional leading timestamp: "06-19-2026 10:00:00"
_TS_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")


def _cpu_avg(line: str) -> float | None:
    """Average per-core load %, ignoring 'off' cores."""
    m = _CPU_RE.search(line)
    if not m:
        return None
    cores = [int(c) for c in _CPU_CORE_RE.findall(m.group(1))]
    return round(sum(cores) / len(cores), 1) if cores else None


def parse_line(line: str, index: int) -> dict | None:
    """Parse one tegrastats line into the CSV schema; None if not a data line."""
    if "RAM" not in line:
        return None

    rails = {name: int(mw) for name, mw in _RAIL_RE.findall(line)}
    temps = {name.lower(): float(v) for name, v in _TEMP_RE.findall(line)}
    ram = _RAM_RE.search(line)
    gpu = _GPU_RE.search(line)
    ts = _TS_RE.search(line)

    # Combined CPU+GPU rail on Orin → vdd_gpu_mw; separate rails preferred.
    vdd_cpu = rails.get("VDD_CPU")
    vdd_gpu = rails.get("VDD_GPU") or rails.get("VDD_CPU_GPU_CV")

    return {
        "t": ts.group(1) if ts else index,
        "cpu_avg_pct": _cpu_avg(line),
        "gpu_pct": int(gpu.group(1)) if gpu else None,
        "ram_used_mb": int(ram.group(1)) if ram else None,
        "vdd_in_mw": rails.get("VDD_IN") or rails.get("VIN_SYS_5V0"),
        "vdd_cpu_mw": vdd_cpu,
        "vdd_gpu_mw": vdd_gpu,
        "vdd_soc_mw": rails.get("VDD_SOC"),
        "gpu_temp_c": temps.get("gpu"),
        "cpu_temp_c": temps.get("cpu"),
    }


def _summary(rows: list[dict]) -> None:
    """Print mean / p50 / p95 / max per numeric column."""
    numeric = [c for c in _COLUMNS if c != "t"]
    print(f"\n{'column':<14}{'mean':>10}{'p50':>10}{'p95':>10}{'max':>10}")
    print("-" * 54)
    for col in numeric:
        vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
        if not vals:
            print(f"{col:<14}{'(no data)':>40}")
            continue
        s = sorted(vals)
        p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
        print(
            f"{col:<14}"
            f"{statistics.mean(vals):>10.1f}"
            f"{statistics.median(vals):>10.1f}"
            f"{p95:>10.1f}"
            f"{max(vals):>10.1f}"
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Parse tegrastats.log → utilization.csv")
    parser.add_argument(
        "logfile",
        nargs="?",
        default=os.environ.get("TEGRASTATS_LOG", "tegrastats.log"),
        help="tegrastats log path (env: TEGRASTATS_LOG)",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("UTILIZATION_CSV", "utilization.csv"),
        help="output CSV path (env: UTILIZATION_CSV)",
    )
    parser.add_argument("--no-summary", action="store_true", help="skip the stats table")
    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.is_file():
        print(f"找不到 tegrastats log：{log_path}")
        raise SystemExit(1)

    rows: list[dict] = []
    for i, line in enumerate(log_path.read_text(errors="ignore").splitlines()):
        row = parse_line(line, i)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"{log_path} 內沒有可解析的 tegrastats 資料行。")
        raise SystemExit(1)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"解析 {len(rows)} 筆 → {out_path}")
    if not args.no_summary:
        _summary(rows)


if __name__ == "__main__":
    main()
