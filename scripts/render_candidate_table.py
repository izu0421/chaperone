"""Render the per-candidate table on the landing page from the data, never by hand.

Every candidate gets three things the page did not carry before:

  WHY IT STANDS OR FALLS, in the words of the test that decided it, not a
  summary of a summary. A rejection with no reason is indistinguishable from a
  rejection with a bad reason.

  WHERE IT CAME FROM. Each row links to the row in the triage report that
  produced its verdict, by anchor, so the rationale behind the call is one click
  away and cannot drift from it. The clashfloor run is a live example of why
  that matters: a confirmed candidate sat in a report no table referenced for a
  day.

  THE FOLD ITSELF, where a structure was published, as a deep link into the
  viewer.

The block is written between markers in index.html, so re-running this replaces
it and nothing is retyped.

Usage:
    python scripts/render_candidate_table.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SM = Path("/data/yzy21/yy/space_agent/structural_modelling")
VIEWER = "reports/protein_pae_2026-11-24_v2/structures"
START = "<!-- CANDIDATE-TABLE-START -->"
END = "<!-- CANDIDATE-TABLE-END -->"

TIER_LABEL = {
    "1": ("Stands", "#2166AC"),
    "2a": ("Partly tested", "#5B8CB8"),
    "2b": ("Weakened", "#C98A93"),
    "2c": ("Gate only", "#9A9AA2"),
    "3": ("Refuted", "#B2182B"),
}


def esc(s) -> str:
    return html.escape(str(s))


def reason_for(r) -> str:
    """Why this candidate stands or falls, from the tests that decided it."""
    bits = []
    for col, name in (("ecd_test", "receptor ectodomain"),
                      ("mature_test", "mature chain"),
                      ("p_test", "partner ectodomain")):
        v = str(getattr(r, col, ""))
        if v.startswith("FAIL"):
            bits.append(f"<b>{name}</b>: {esc(v.replace('FAIL -- ', ''))}")
        elif v == "PASS":
            bits.append(f"<b>{name}</b>: survives, with its own controls")
        elif v == "weakened":
            bits.append(f"<b>{name}</b>: interface weakens to {getattr(r, 'pae_mature', float('nan')):.2f} &#8491;")
        elif v.startswith("WITHHELD"):
            bits.append(f"<b>{name}</b>: {esc(v.replace('WITHHELD -- ', 'withheld, '))}")
    kind = str(getattr(r, "interface", ""))
    if kind == "IMPOSSIBLE cytoplasmic":
        bits.insert(0, "<b>physics</b>: a secreted partner placed on the cytoplasmic face, "
                       "which it has no route to")
    if not bits:
        bits.append("no construct test has returned yet; the gate is the only evidence")
    return "; ".join(bits) + "."


def main() -> None:
    t = pd.read_csv(SM / "results" / "candidate_ranking.csv")
    v = pd.read_csv(ROOT / "reports" / "screen_verdicts_current" / "verdicts.csv")
    run_of = {(str(r.protein_b), str(r.protein_a)): str(r.run) for r in v.itertuples()}
    rat_of = {(str(r.protein_b), str(r.protein_a)): str(r.rationale or "")
              for r in v.itertuples()}
    sheet = pd.read_csv("/data/yzy21/yy/space_agent/lr_atlas/bench/af3_orphan_sheet_v2.csv")
    acc = dict(zip(sheet.receptor_gene, sheet.receptor_uniprot))
    acc.update(dict(zip(sheet.partner_gene, sheet.partner_uniprot)))
    have_pdb = {p.stem for p in (ROOT / VIEWER / "s").glob("*.pdb")}

    rows = []
    for r in t.sort_values(["tier", "pae_screen"]).itertuples():
        tier = str(r.tier).split(" ")[0]
        label, colour = TIER_LABEL.get(tier, (tier, "#9A9AA2"))
        run = run_of.get((r.receptor, r.partner), "")
        links = []
        if run and (ROOT / "reports" / run / "report.html").exists():
            links.append(f'<a href="reports/{esc(run)}/report.html#pair-'
                         f'{esc(r.partner)}-{esc(r.receptor)}">triage</a>')
        job = f"{r.receptor}__{r.partner}"
        if job in have_pdb:
            links.append(f'<a href="{VIEWER}/#{esc(job)}">fold</a>')
        for g in (r.receptor, r.partner):
            if g in acc:
                links.append(f'<a href="https://www.uniprot.org/uniprotkb/{esc(acc[g])}">'
                             f'{esc(g)}</a>')
        first = rat_of.get((r.receptor, r.partner), "").split(". ")[0]
        rows.append(f"""<tr>
  <td><b>{esc(r.receptor)} + {esc(r.partner)}</b><br>
      <span class="mut">{esc(r.pae_screen):.4s} &#8491; &middot; {esc(r.interface)}</span></td>
  <td><span class="badge" style="background:{colour}">{esc(label)}</span></td>
  <td class="why">{reason_for(r)}
      <div class="mut">Triaged in <code>{esc(run)}</code>. {esc(first)}.</div></td>
  <td class="lnk">{' &middot; '.join(links)}</td>
</tr>""")

    block = f"""{START}
<h2>Every candidate, and what decided it</h2>
<p>All {len(t)} pairs the agent returned as confirmed, with the construct test that settled
each one and a link to the triage row that produced its verdict. A rejection with no
reason attached is indistinguishable from a rejection with a bad reason, and a verdict
whose source is not linked drifts from it &mdash; which is how a confirmed candidate spent
a day in a report no table referenced.</p>
<table class="cands">
<tr><th>Pair</th><th>Standing</th><th>What decided it</th><th>Source</th></tr>
{chr(10).join(rows)}
</table>
{END}"""

    page = ROOT / "index.html"
    s = page.read_text()
    if START in s and END in s:
        s = s[:s.index(START)] + block + s[s.index(END) + len(END):]
    else:
        s = s.replace("<h2>Code</h2>", block + "\n\n<h2>Code</h2>", 1)
    page.write_text(s)
    print(f"rendered {len(rows)} candidates into index.html")
    print(t.tier.value_counts().to_string())


if __name__ == "__main__":
    main()
