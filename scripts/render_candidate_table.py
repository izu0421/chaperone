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
S_START = "<!-- SURVIVORS-START -->"
S_END = "<!-- SURVIVORS-END -->"

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

    # ---- the two blocks the page leads with --------------------------------
    passes = {}
    for r in t.itertuples():
        got = [c for c in ("ecd_test", "mature_test", "p_test")
               if str(getattr(r, c, "")) == "PASS"]
        fails = [c for c in ("ecd_test", "mature_test", "p_test")
                 if str(getattr(r, c, "")).startswith("FAIL")]
        if got and not fails:
            passes[f"{r.receptor}+{r.partner}"] = (r, got)

    EVIDENCE = {
        "GPR158+NMS": [
            ("Gate", "2.69 &#8491; interface PAE on the intact receptor"),
            ("Where it sits", "97% of the receptor side of the interface is extracellular"),
            ("Truncation", "on the receptor&rsquo;s N-terminal domain alone, 24&ndash;417, with no "
                           "stitched loops: <b>2.45 &#8491;</b> against decoys at 9.66 (RNase&nbsp;1), "
                           "12.15 (lysozyme) and 22.66 (albumin), 25 models each"),
            ("Mature chain", "2.69 &rarr; 3.01 &#8491; with the signal peptide removed, on a cut "
                             "that improves AlphaFold3 on 24 chemokine pairs whose answer is known"),
            ("Prior record", "STRING nothing, CellPhoneDB nothing, PubMed co-mentions 0"),
        ],
        "MRGPRX2+SPX": [
            ("Gate", "3.51 &#8491; interface PAE, the weakest gate value of the three"),
            ("Where it sits", "extracellular plus transmembrane with no cytoplasmic contacts, "
                              "which is where a peptide agonist is supposed to bind"),
            ("Mature chain", "survives the cut at 3.51 &#8491;, under the gate"),
            ("Why it is thinner than the other two", "its only test is the mature-chain cut, "
                 "which has a panel-level control (12 of 12 known binders keep their interface) "
                 "but no negative control for this pair. The other two each beat decoys or "
                 "permuted negatives chosen to match them. MRGPRX2 is also a famously "
                 "promiscuous receptor, so a peptide in its pocket is a low bar"),
        ],
        "GPR83+GRIA4": [
            ("Gate", "3.01 &#8491; interface PAE"),
            ("Where it sits", "94% extracellular on <b>both</b> chains, the only ecto&ndash;ecto pair "
                              "in the shortlist"),
            ("Partner ectodomain", "cutting GRIA4 to its ectodomain <b>improves</b> it, 3.01 &rarr; "
                                   "<b>2.21 &#8491;</b>, against a best permuted negative of 4.96 &#8491; "
                                   "(the SLITRK2 ectodomain on the same receptor)"),
            ("Open question", "GRIA4 is an obligate tetramer and it was folded as a lone chain, so "
                              "the surface GPR83 uses may be the dimer interface. The same construct "
                              "with a second ectodomain copy present is folding now, alongside a "
                              "control that checks the two copies pair at all"),
        ],
    }

    cards = []
    for name, (r, got) in passes.items():
        ev = EVIDENCE.get(name)
        if not ev:
            ev = [("Passed", ", ".join(c.replace("_test", "") for c in got))]
        li = "".join(f"<li><b>{esc(k)}.</b> {v}</li>" for k, v in ev)
        cards.append(f"""<div class="hit">
  <h3>{esc(name.replace('+', ' + '))}</h3>
  <ul>{li}</ul>
</div>""")

    buckets = {}
    for r in t.itertuples():
        why = None
        for c in ("ecd_test", "mature_test", "p_test"):
            v = str(getattr(r, c, ""))
            if v.startswith("FAIL"):
                why = v.replace("FAIL -- ", ""); break
        if why is None:
            continue
        if "cytoplasmic face, partner is secreted" in why:
            k = ("A secreted partner placed on the cytoplasmic face",
                 "It has no route to the cytosol, so the pose cannot exist.")
        elif "cytoplasmic" in why:
            k = ("The receptor&rsquo;s cytoplasmic face, after the partner&rsquo;s tail was removed",
                 "Cutting the partner removes the partner&rsquo;s impossible half and leaves the "
                 "receptor&rsquo;s. Their permuted negatives sit on the same face, so the margin "
                 "compares two poses that cannot exist.")
        elif "permuted ectodomain" in why:
            k = ("Beaten by another candidate&rsquo;s ectodomain",
                 "The same receptor docks an unrelated, size-matched ectodomain at least as well.")
        elif "interface lost" in why:
            k = ("The interface goes when the signal peptide goes",
                 "Both chains still fold; only the interface is lost.")
        else:
            k = (why.capitalize(), "")
        buckets.setdefault(k, []).append(f"{r.receptor} + {r.partner}")

    rej = "".join(
        # k[0] already contains HTML entities written by hand; esc() would turn
        # &rsquo; into &amp;rsquo; and the apostrophe would render as literal text
        f"<tr><td><b>{k[0]}</b><br><span class=\"mut\">{k[1]}</span></td>"
        f"<td>{len(v)}</td><td class=\"why\">{esc(', '.join(sorted(v)))}</td></tr>"
        for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1])))

    survivors = f"""{S_START}
<h2>What survived</h2>
<p>{len(passes)} of the {len(t)} confirmed pairs are still standing: each has at least one
construct test it passed and none that it failed. They are not equally well supported, and
the cards say where each one is thin. What they have in common is that the test could have
gone the other way, which is the only kind of evidence this screen has produced that is
worth anything.</p>
{"".join(cards)}
<p class="mut" style="font-size:12px">Both are AlphaFold3 predictions with no experimental
support of any kind. The method&rsquo;s own ceiling bounds them: AUROC 0.519 on
degree-balanced HuRI. These controls show the two are not the artefacts the other
{len(t) - len(passes)} are. They do not show the interactions are real.</p>

<h2>Why the others fell</h2>
<table>
<tr><th>reason</th><th>n</th><th>pairs</th></tr>
{rej}
</table>
<p>The rest are not refuted and not confirmed: their construct test is built and folding,
or the pair fails at full length on the re-fold arm&rsquo;s own alignment, in which case
cutting it attributes nothing in either direction.</p>
{S_END}"""

    block = f"""{START}
<h2>Every candidate, and what decided it</h2>
<p>All {len(t)} pairs the agent returned as confirmed, with the construct test that settled
each one and a link to the triage row that produced its verdict. A rejection with no
reason attached is indistinguishable from a rejection with a bad reason, and a verdict
whose source is not linked drifts from it. That is how a confirmed candidate spent a day in
a report no table referenced.</p>
<table class="cands">
<tr><th>Pair</th><th>Standing</th><th>What decided it</th><th>Source</th></tr>
{chr(10).join(rows)}
</table>
{END}"""

    page = ROOT / "index.html"
    s = page.read_text()
    if S_START in s and S_END in s:
        s = s[:s.index(S_START)] + survivors + s[s.index(S_END) + len(S_END):]
    else:
        s = s.replace('<h2>Re-folding the shortlist as molecules the cell could make</h2>',
                      survivors + '\n\n<h2>Re-folding the shortlist as molecules the cell '
                      'could make</h2>', 1)
    if START in s and END in s:
        s = s[:s.index(START)] + block + s[s.index(END) + len(END):]
    else:
        s = s.replace("<h2>Code</h2>", block + "\n\n<h2>Code</h2>", 1)
    page.write_text(s)
    print(f"rendered {len(rows)} candidates, {len(passes)} survivors, "
          f"{sum(len(v) for v in buckets.values())} rejected into index.html")
    print(t.tier.value_counts().to_string())


if __name__ == "__main__":
    main()
