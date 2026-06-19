#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai, Yanting Lin
# Tatung University — I4210 AI實務專題
"""tools/md2pdf.py — render a Markdown file to PDF (CJK + tables) via headless Chromium.

    python tools/md2pdf.py report/FINAL_REPORT.md report/FINAL_REPORT.pdf
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

_CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Noto Serif CJK TC","AR PL UMing TW",serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 15pt; border-bottom: 1px solid #999; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 12pt; margin-top: 14px; }
h1,h2,h3 { page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.3pt; }
th,td { border: 1px solid #bbb; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eef0f2; }
code { font-family: "DejaVu Sans Mono",monospace; background: #f4f4f4; padding: 1px 3px; font-size: 9pt; }
pre { background: #f6f8fa; border: 1px solid #ddd; padding: 6px; overflow: hidden;
      font-family: "DejaVu Sans Mono",monospace; font-size: 7pt; line-height: 1.25; white-space: pre; }
pre code { background: none; padding: 0; font-size: inherit; font-family: inherit; }
blockquote { border-left: 3px solid #ccc; margin: 8px 0; padding: 2px 10px; color: #444; background: #fafafa; }
table, pre, blockquote { page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
"""


def main() -> None:
    """Convert argv[1] (markdown) to argv[2] (pdf)."""
    if len(sys.argv) != 3:
        print("usage: md2pdf.py <input.md> <output.pdf>")
        raise SystemExit(2)
    md_path, pdf_path = Path(sys.argv[1]), Path(sys.argv[2])
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>{html_body}</body></html>"

    html_path = md_path.with_suffix(".tmp.html")
    html_path.write_text(html, encoding="utf-8")
    user_dir = Path.home() / ".cache" / "md2pdf-chrome"
    user_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(  # noqa: S603 — fixed chromium invocation
            [
                "chromium",
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={user_dir}",
                f"--print-to-pdf={pdf_path}",
                str(html_path),
            ],
            check=True,
            timeout=120,
            capture_output=True,
        )
    finally:
        html_path.unlink(missing_ok=True)

    if pdf_path.is_file():
        print(f"✅ {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    else:
        print("❌ PDF 未產生")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
