"""Retro-fit per-pair anchors into reports that were built before build_report.py had them.

The landing page links to the row that explains each verdict. A report with no
anchors forces the explanation to be retyped on the landing page instead, where
it drifts from the run that produced it.

Idempotent: a row that already carries an id is left alone.

Usage:
    python scripts/add_report_anchors.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r'<tr>\s*\n\s*<td><b>([^<]+?)\s*&harr;\s*([^<]+?)</b></td>')


def main() -> None:
    for rep in sorted(ROOT.glob("reports/*/report.html")):
        s = rep.read_text()
        n = 0

        def sub(m: re.Match) -> str:
            nonlocal n
            a, b = m.group(1).strip(), m.group(2).strip()
            n += 1
            anchor = f"pair-{a}-{b}".replace(" ", "_")
            return m.group(0).replace("<tr>", f'<tr id="{anchor}">', 1)

        out = ROW.sub(sub, s)
        if n:
            rep.write_text(out)
        print(f"  {rep.parent.name:38s} {n} pair anchors")


if __name__ == "__main__":
    main()
