"""Build a single self-contained HTML report from a verdicts CSV + the
per-candidate log/*.json transcripts, so every conclusion is traceable back
to the raw tool evidence it was based on (not just the model's paraphrase).

Usage:
    python -m chaperone.build_report data/verdicts_full.csv log --out data/report.html
"""
import argparse
import csv
import html
import json
import sys
from pathlib import Path

from .strategy_utils import normalize_str_list  # noqa: E402
from .paths import PROJECT_ROOT  # noqa: E402
from .deterministic_gate import fold_evidence_status  # noqa: E402

VERDICT_COLORS = {
    "CONFIRMED_NOVEL": "#2EC4B6",
    "LIKELY_SUBCOMPLEX": "#5B8DEF",
    "ALREADY_KNOWN": "#8A8A8A",
    "LIKELY_ARTIFACT_PTM": "#FF9F1C",
    "IMPLAUSIBLE": "#E05D5D",
    "INSUFFICIENT_EVIDENCE": "#B0B7C3",
}

EVIDENCE_FIELDS = [
    ("coexpression_evidence", "Coexpression"),
    ("subcellular_evidence", "Subcellular location"),
    ("ptm_glycosylation_evidence", "PTM / glycosylation"),
    ("cooccurrence_evidence", "PubMed co-occurrence"),
    ("known_interaction_evidence", "STRING / CellPhoneDB"),
]

CSS = """
:root { --navy:#1B2A4A; --teal:#2EC4B6; --gold:#FF9F1C; --grey:#6B7280; --light:#F4F6F8; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Calibri, Helvetica, Arial, sans-serif; margin:0; background:#fff; color:var(--navy); }
header { background:var(--navy); color:#fff; padding:28px 40px; }
header h1 { margin:0 0 6px 0; font-size:28px; }
header p { margin:0; color:var(--teal); font-size:14px; }
.summary { display:flex; gap:16px; padding:24px 40px; flex-wrap:wrap; border-bottom:1px solid #eee; }
.stat { background:var(--light); border-radius:10px; padding:14px 20px; min-width:120px; text-align:center; }
.stat .n { font-size:26px; font-weight:700; color:var(--navy); }
.stat .l { font-size:12px; color:var(--grey); text-transform:uppercase; letter-spacing:.04em; }
main { padding: 10px 40px 60px; }
.card { border:1px solid #e3e6ea; border-radius:12px; margin:18px 0; overflow:hidden; }
.card-head { display:flex; align-items:center; gap:14px; padding:14px 18px; cursor:pointer; background:#fff; }
.card-head:hover { background:var(--light); }
.pair { font-size:17px; font-weight:700; }
.badge { color:#fff; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.02em; }
.conf { font-size:12px; color:var(--grey); border:1px solid #ddd; border-radius:999px; padding:3px 10px; }
.metrics { font-size:12px; color:var(--grey); margin-left:auto; }
.card-body { padding: 4px 18px 18px; border-top:1px solid #eee; display:none; }
.card.open .card-body { display:block; }
.field { margin:10px 0; }
.field .k { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--grey); font-weight:700; }
.field .v { font-size:14px; line-height:1.5; margin-top:2px; }
.followup { background:#FFF6E8; border-left:3px solid var(--gold); padding:10px 14px; border-radius:6px; margin-top:12px; }
.executed { background:#EAFBF8; border-left:3px solid var(--teal); padding:10px 14px; border-radius:6px; margin-top:12px; }
.executed .tag { display:inline-block; background:var(--teal); color:#fff; font-size:10px; font-weight:700; padding:2px 8px; border-radius:999px; margin-right:8px; letter-spacing:.04em; }
.executed .metric { font-family:monospace; font-weight:700; }
details.tools { margin-top:14px; }
.validation-strategy { background:#F4F0FF; border-left:3px solid #6C4EE0; padding:10px 14px; border-radius:6px; margin-top:12px; }
.validation-strategy summary { cursor:pointer; font-weight:600; color:#6C4EE0; }
.validation-strategy .field { margin-top:8px; }
details.tools summary { cursor:pointer; font-weight:700; font-size:13px; color:var(--navy); padding:6px 0; }
.toolcall { background:var(--light); border-radius:8px; padding:10px 14px; margin:8px 0; }
.toolcall .name { font-family: monospace; font-weight:700; color:var(--navy); }
.toolcall .err { color:var(--grey); font-weight:700; }
.toolcall pre { white-space:pre-wrap; word-break:break-word; font-size:12px; background:#fff; border:1px solid #e3e6ea; border-radius:6px; padding:8px; margin:6px 0 0; max-height:260px; overflow:auto; }
.struct-toggle { display:inline-block; margin-top:8px; padding:5px 12px; font-size:12px; font-weight:700; color:var(--navy); background:#fff; border:1px solid var(--teal); border-radius:999px; cursor:pointer; }
.struct-toggle:hover { background:var(--teal); color:#fff; }
.viewer-container { margin-top:10px; }
.viewer { width:100%; height:420px; border:1px solid #e3e6ea; border-radius:8px; position:relative; background:#fafbfc; }
.viewer-legend { font-size:11px; color:var(--grey); margin-top:4px; }
.viewer-legend .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }
.validation { margin:0 40px 10px; padding:20px 24px; background:var(--light); border-radius:12px; }
.validation h2 { margin:0 0 6px; font-size:18px; color:var(--navy); }
.validation-note { margin:0 0 14px; font-size:13px; color:var(--grey); line-height:1.5; }
.validation-table { width:100%; border-collapse:collapse; font-size:13px; }
.validation-table th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--grey); padding:6px 10px; border-bottom:2px solid #ddd; }
.validation-table td { padding:8px 10px; border-bottom:1px solid #e3e6ea; vertical-align:top; }
.validation-table tr:hover td { background:#fff; }
.rejected-group { margin-top:10px; background:#fff; border-radius:8px; padding:2px 12px; }
.rejected-group summary { cursor:pointer; padding:10px 0; font-size:13px; color:var(--navy); font-weight:600; }
.rejected-group summary .badge { margin-right:8px; }
.rejected-group table { margin-bottom:10px; }
"""

JS_TEMPLATE = """
document.querySelectorAll('.card-head').forEach(h => {
  h.addEventListener('click', () => h.closest('.card').classList.toggle('open'));
});

const STRUCTURES = __STRUCTURES_JSON__;
const _viewers = {};

function toggleStructure(id) {
  const container = document.getElementById('viewer-wrap-' + id);
  const btn = document.getElementById('struct-btn-' + id);
  if (container.style.display === 'none' || !container.style.display) {
    container.style.display = 'block';
    btn.textContent = 'Hide 3D structure';
    if (!_viewers[id]) initViewer(id);
  } else {
    container.style.display = 'none';
    btn.textContent = 'Show 3D structure';
  }
}

function initViewer(id) {
  const el = document.getElementById('viewer-' + id);
  const s = STRUCTURES[id];
  if (!s) {
    el.innerHTML = "<p style='color:#E05D5D;padding:12px;'>No structure data found for this candidate (internal error — id '" + id + "' missing from STRUCTURES).</p>";
    return;
  }
  if (typeof $3Dmol === 'undefined') {
    el.innerHTML = "<p style='color:#E05D5D;padding:12px;'>3D viewer library (3Dmol.js, loaded from 3dmol.org) failed to load — likely no internet access to that domain from wherever this page is being viewed. The structure data itself IS in this file; only the in-browser viewer needs that external script.</p>";
    return;
  }
  try {
    const viewer = $3Dmol.createViewer(el, {backgroundColor: 'white'});
    viewer.addModel(s.cif, 'cif');
    viewer.setStyle({}, {cartoon: {colorscheme: 'chainHetatm'}});
    if (s.interface_a && s.interface_a.length) {
      viewer.setStyle({chain: s.chain_a, resi: s.interface_a}, {cartoon: {color: '#FF9F1C'}});
    }
    if (s.interface_b && s.interface_b.length) {
      viewer.setStyle({chain: s.chain_b, resi: s.interface_b}, {cartoon: {color: '#FF9F1C'}});
    }
    // Label each chain with its actual gene/protein name at its centroid —
    // "chain A/B" means nothing on its own; this is the fix for the
    // user-reported "label the subunits" request.
    if (s.chain_labels) {
      const model = viewer.getModel(0);
      Object.keys(s.chain_labels).forEach(function(chainId) {
        const atoms = model.selectedAtoms({chain: chainId});
        if (!atoms.length) return;
        let sx = 0, sy = 0, sz = 0;
        atoms.forEach(function(a) { sx += a.x; sy += a.y; sz += a.z; });
        const n = atoms.length;
        viewer.addLabel(s.chain_labels[chainId], {
          position: {x: sx / n, y: sy / n, z: sz / n},
          backgroundColor: '#1B2A4A', backgroundOpacity: 0.85,
          fontColor: 'white', fontSize: 14, borderThickness: 0,
        });
      });
    }
    viewer.zoomTo();
    viewer.render();
    _viewers[id] = viewer;
  } catch (err) {
    el.innerHTML = "<p style='color:#E05D5D;padding:12px;'>3D viewer failed to render: " + String(err && err.message || err) + "</p>";
  }
}
"""


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def render_tool_calls(tool_calls):
    if not tool_calls:
        return "<p style='color:var(--grey);font-size:13px;'>No tool-call log captured for this candidate (older run, before traceability logging was added).</p>"
    parts = []
    for i, call in enumerate(tool_calls, 1):
        err = " — <span class='err'>ERROR</span>" if call.get("is_error") else ""
        output = call.get("output")
        output_str = json.dumps(output, indent=2) if isinstance(output, (dict, list)) else str(output)
        parts.append(
            f"<div class='toolcall'>"
            f"<div><span class='name'>{i}. {esc(call.get('tool'))}</span>{err} "
            f"<span style='color:var(--grey);font-size:12px;'>({call.get('elapsed_seconds', '?')}s)</span></div>"
            f"<div style='font-size:12px;color:var(--grey);margin-top:4px;'>input: {esc(json.dumps(call.get('input')))}</div>"
            f"<pre>{esc(output_str)}</pre>"
            f"</div>"
        )
    return "".join(parts)


VALIDATION_TIERS = {
    "CONFIRMED_NOVEL": {
        "label": "Top pick — novel & plausible",
        "color": "#2EC4B6",
        "reason": "No known/curated hit anywhere, and expression/localization support a direct interaction — the strongest class of untested hypothesis here.",
    },
    "LIKELY_SUBCOMPLEX": {
        "label": "Validate the completed complex, not the pair",
        "color": "#5B8DEF",
        "reason": "Likely a fragment of a larger complex — validate together with the named subunit(s), not the binary pair alone.",
    },
    "LIKELY_ARTIFACT_PTM": {
        "label": "Resolve PTM/glycosylation concern first",
        "color": "#FF9F1C",
        "reason": "May be a modeling artifact (unmodeled glycan/PTM at the interface) — check structurally (fold_complex) before committing wet-lab time.",
    },
}


INTERFACE_PLDDT_CONFIDENT = 50  # generic "not garbage" bar for an AF3 interface


def classify_validation(row: dict, executed_results: list) -> dict:
    """Return {tier, label, reason, note} for candidates worth prioritizing
    for experimental validation. A verdict only *names* an open question
    (missing subunit, possible PTM/glycosylation artifact) — it is not
    itself grounds to recommend wet-lab time. LIKELY_SUBCOMPLEX and
    LIKELY_ARTIFACT_PTM therefore only return Yes once that specific
    question has actually been checked and resolved by a real executed fold,
    not merely proposed as a follow-up. Returns None otherwise, and for
    excluded verdicts (ALREADY_KNOWN: already proven; IMPLAUSIBLE: unlikely
    to be real; INSUFFICIENT_EVIDENCE: not enough evidence to prioritize
    either way — all listed separately, not as a validation pick)."""
    verdict = row.get("verdict")
    tier = VALIDATION_TIERS.get(verdict)
    if not tier:
        return None

    note = None

    # A real fold can clear pLDDT/PTM checks numerically while its own
    # topology finding (an interface spanning BOTH Cytoplasmic AND
    # Extracellular residues — impossible across an intact membrane, per
    # fold_evidence_status) argues the modeled complex can't be trusted at
    # all. Reuse the gate's own read of this same fold data rather than a
    # second, cruder copy of the logic — an earlier version of this function
    # duplicated a pLDDT-only check that missed exactly this: KLRK1/KLRD1's
    # fold cleared pLDDT>=50 on every interface yet fold_evidence_status
    # independently flagged topology_violation on the same data (KLRD1's
    # real, disulfide-confirmed partner is KLRC1, not KLRK1).
    fold_status = fold_evidence_status(executed_results)
    blocked_by_topology = fold_status["status"] == "topology_violation"

    if verdict == "LIKELY_SUBCOMPLEX":
        if not row.get("other_subunits"):
            return None  # no concrete subunit(s) even named - nothing to resolve against
        if blocked_by_topology:
            return None  # the fold itself argues against this modeled complex, regardless of interface pLDDT
        interfaces = [
            iface
            for r in executed_results
            for iface in (r.get("structural_analysis") or {}).get("interfaces", [])
        ]
        confirmed = interfaces and all(
            (iface.get("interface_plddt_a") or {}).get("mean", 0) >= INTERFACE_PLDDT_CONFIDENT
            and (iface.get("interface_plddt_b") or {}).get("mean", 0) >= INTERFACE_PLDDT_CONFIDENT
            for iface in interfaces
        )
        if not confirmed:
            return None  # proposed subunit(s) not yet re-folded and confirmed
        note = f"Re-folded with {row['other_subunits']}: confident interface confirmed (pLDDT >= {INTERFACE_PLDDT_CONFIDENT})."

    if verdict == "LIKELY_ARTIFACT_PTM":
        # Only resolved if this was actually structurally checked
        # (fold_complex + interface analysis) and no PTM/glycosylation site
        # landed on the modeled interface. A candidate whose mechanism is
        # fundamentally glycan-mediated (e.g. a galectin) isn't something a
        # single fold can clear regardless — it correctly stays unresolved.
        if blocked_by_topology:
            return None  # a stronger, more fundamental structural problem than the named PTM concern
        resolved = False
        for r in executed_results:
            for iface in (r.get("structural_analysis") or {}).get("interfaces", []):
                has_ptm = (iface.get("ptm_sites_at_interface_a") or []) or (iface.get("ptm_sites_at_interface_b") or [])
                if not has_ptm:
                    resolved = True
                    note = "Checked structurally: no known PTM/glycosylation site at the modeled interface — concern resolved."
                    break
            if resolved:
                break
        if not resolved:
            return None  # PTM/glycosylation concern not yet checked/cleared

    return {
        "tier": verdict,
        "label": tier["label"],
        "color": tier["color"],
        "reason": tier["reason"],
        "note": note,
    }


def load_validation_strategies(path: Path) -> dict:
    if not path.exists():
        return {}
    strategies = json.loads(path.read_text())
    return {f"{s['protein_a']}/{s['protein_b']}": s for s in strategies}


METHOD_LABELS = {
    "PLA": "PLA (co-localization)",
    "stimulation_assay": "Stimulation assay",
    "co_ip": "Co-IP",
}


REJECTED_VERDICT_BLURBS = {
    "IMPLAUSIBLE": "No expression/localization/structural support for a direct interaction.",
    "ALREADY_KNOWN": "Already proven in the literature/curated databases; not a new finding.",
    "INSUFFICIENT_EVIDENCE": "Data was genuinely too thin to call either way — not rejected, just unresolved.",
    "LIKELY_SUBCOMPLEX": "Likely an incomplete complex — proposed subunit(s) not yet re-folded and confirmed.",
    "LIKELY_ARTIFACT_PTM": "Possible modeling artifact (unmodeled glycan/PTM at the interface) — not yet resolved structurally.",
}


def render_validation_summary(enriched_rows: list, strategies: dict = None) -> str:
    """One flat table, every candidate, answering exactly one question per
    row: is this worth validating experimentally, yes or no, and why. Not
    split into a "recommended" table plus collapsed "rejected" sections —
    that structure was reported as not what was actually asked for."""
    strategies = strategies or {}

    def sort_key(row_v):
        row, v = row_v
        tier_order = {"CONFIRMED_NOVEL": 0, "LIKELY_SUBCOMPLEX": 1, "LIKELY_ARTIFACT_PTM": 2}
        worth_rank = 0 if v else 1
        return (worth_rank, tier_order.get(v["tier"], 9) if v else 0, -_safe_float(row.get("iptm")))

    entries = [(row, classify_validation(row, ex)) for row, ex in enriched_rows]
    entries.sort(key=sort_key)

    rows_html = ""
    n_yes = 0
    for row, v in entries:
        pair = f"{esc(row['protein_a'])} &harr; {esc(row['protein_b'])}"
        if v:
            n_yes += 1
            worth_badge = "<span class='badge' style='background:#2EC4B6;'>Yes</span>"
            why = esc(v["note"] or v["reason"])
            strategy = strategies.get(f"{row['protein_a']}/{row['protein_b']}")
            if strategy:
                method = METHOD_LABELS.get(strategy["method"], esc(strategy["method"]))
                tissues = ", ".join(normalize_str_list(strategy.get("target_tissues")))
                why += f"<br><span style='color:var(--grey);'>Suggested: <b>{esc(method)}</b> in {esc(tissues)}</span>"
        else:
            worth_badge = "<span class='badge' style='background:#B0B7C3;'>No</span>"
            blurb = REJECTED_VERDICT_BLURBS.get(row.get("verdict"), row.get("verdict") or "")
            first_sentence = (row.get("rationale") or "").split(". ")[0]
            why = esc(blurb) + (f" {esc(first_sentence)}." if first_sentence else "")
        rows_html += f"""
        <tr>
          <td><b>{pair}</b></td>
          <td>{worth_badge}</td>
          <td style="font-size:12px;color:var(--grey);">{why}</td>
        </tr>"""

    return f"""
    <section class="validation">
      <h2>Worth validating experimentally?</h2>
      <p class="validation-note">{n_yes} of {len(entries)} candidates are not already proven in the literature, and
      any open structural/mechanism concern raised has actually been checked and cleared — every candidate is
      listed, nothing hidden.</p>
      <table class="validation-table">
        <thead><tr><th>Pair</th><th>Worth validating?</th><th>Why</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </section>
    """


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_executed_follow_ups(fold_summary_path: Path) -> dict:
    """Map 'PROTEIN_A/PROTEIN_B' -> list of executed fold-job results, keyed
    by the `source_pair` field written by chaperone.run_follow_up_folds."""
    if not fold_summary_path.exists():
        return {}
    by_pair = {}
    for result in json.loads(fold_summary_path.read_text()):
        by_pair.setdefault(result.get("source_pair"), []).append(result)
    return by_pair


def render_structure_toggle(result: dict, viewer_id: str, structures: dict) -> str:
    """Register the model .cif (if it still exists on disk) into `structures`
    for embedding, and return the toggle button + lazy-init viewer container.
    Returns "" if there's no readable structure to show."""
    cif_path = result.get("model_cif")
    if not cif_path or not Path(cif_path).exists():
        return ""
    try:
        cif_text = Path(cif_path).read_text()
    except OSError:
        return ""

    interfaces = (result.get("structural_analysis") or {}).get("interfaces", [])
    iface = interfaces[0] if interfaces else {}
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_labels = {chain_letters[i]: c["identifier"] for i, c in enumerate(result.get("chains", []))}
    structures[viewer_id] = {
        "cif": cif_text,
        "chain_a": iface.get("chain_a", "A"),
        "chain_b": iface.get("chain_b", "B"),
        # full per-residue lists only present in the un-summarized analysis
        # (backfill jobs); agent-triggered live folds only carry counts to
        # save tokens, so highlighting degrades gracefully to chain-only color
        "interface_a": iface.get("interface_residues_a") or [],
        "interface_b": iface.get("interface_residues_b") or [],
        "chain_labels": chain_labels,
    }
    legend_chains = " &middot; ".join(f"<b>{esc(k)}</b> = {esc(v)}" for k, v in chain_labels.items())
    return f"""
    <button class="struct-toggle" id="struct-btn-{esc(viewer_id)}" onclick="toggleStructure('{esc(viewer_id)}')">Show 3D structure</button>
    <div class="viewer-container" id="viewer-wrap-{esc(viewer_id)}" style="display:none;">
      <div class="viewer" id="viewer-{esc(viewer_id)}"></div>
      <div class="viewer-legend">Chains: {legend_chains}</div>
      <div class="viewer-legend"><span class="swatch" style="background:#8A8A8A;"></span>cartoon by chain
        <span class="swatch" style="background:#FF9F1C;margin-left:10px;"></span>modeled interface residues (if available)</div>
    </div>
    """


def render_executed_follow_up(result: dict, viewer_id: str, structures: dict) -> str:
    if result.get("status") != "ok":
        return (
            f"<div class='executed' style='border-color:var(--grey);background:#F4F6F8;'>"
            f"<span class='tag' style='background:var(--grey);'>ATTEMPTED</span>"
            f"Fold job <code>{esc(result.get('job_name'))}</code> failed: {esc(result.get('error'))}"
            f"</div>"
        )
    chains = " + ".join(c["identifier"] for c in result.get("chains", []))
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_names = {chain_letters[i]: c["identifier"] for i, c in enumerate(result.get("chains", []))}
    analysis_html = ""
    for iface in (result.get("structural_analysis") or {}).get("interfaces", []):
        ptm_a = iface.get("ptm_sites_at_interface_a") or []
        ptm_b = iface.get("ptm_sites_at_interface_b") or []
        ptm_note = ""
        if ptm_a or ptm_b:
            descs = [p["description"] for p in [*ptm_a, *ptm_b]]
            ptm_note = f" &mdash; <b style='color:var(--gold);'>PTM/glycosylation site AT interface: {esc('; '.join(descs))}</b>"
        else:
            ptm_note = " &mdash; no known PTM/glycosylation site at this interface"

        topo_a = iface.get("interface_topology_a") or []
        topo_b = iface.get("interface_topology_b") or []
        topo_note = ""
        if topo_a or topo_b:
            combined_set = set(topo_a) | set(topo_b)
            combined = sorted(combined_set)
            # Only flag the genuine physical impossibility — a single modeled
            # contact touching BOTH sides of an intact membrane at once.
            # Cytoplasmic+Transmembrane or Extracellular+Transmembrane alone
            # is normal (real interfaces routinely include TM-helix boundary
            # residues) and was over-firing here before this was narrowed.
            both_sides = "Cytoplasmic" in combined_set and "Extracellular" in combined_set
            style = "color:#E05D5D;font-weight:700;" if both_sides else "color:var(--grey);"
            flag = (
                " (touches BOTH Cytoplasmic and Extracellular residues in one contact — a real membrane "
                "can't be crossed like that. Note this can show up even at high interface pLDDT: AF3 never "
                "models a lipid bilayer, so a confident fold can still place topologically opposite regions "
                "spatially close together with nothing keeping them apart. Worth inspecting the actual 3D "
                "structure below rather than treating this as automatic proof either way)"
                if both_sides else ""
            )
            topo_note = f"<div style='font-size:12px;{style}margin-top:2px;'>Membrane topology at interface: {esc(', '.join(combined))}{flag}</div>"

        label_a = f"{iface['chain_a']} ({chain_names.get(iface['chain_a'], '?')})"
        label_b = f"{iface['chain_b']} ({chain_names.get(iface['chain_b'], '?')})"
        analysis_html += (
            f"<div style='font-size:12px;color:var(--grey);margin-top:4px;'>"
            f"Interface {esc(label_a)}&ndash;{esc(label_b)}: "
            f"{iface['n_interface_residues_a']}/{iface['n_interface_residues_b']} residues, "
            f"interface pLDDT {esc((iface.get('interface_plddt_a') or {}).get('mean'))}/"
            f"{esc((iface.get('interface_plddt_b') or {}).get('mean'))}{ptm_note}"
            f"</div>{topo_note}"
        )
    note = result.get("note") or result.get("reason") or ""
    structure_html = render_structure_toggle(result, viewer_id, structures)
    source = result.get("_source")
    if source == "live_agent":
        source_note = (
            "<b>Live, in-conversation:</b> the agent called fold_complex itself during its own original "
            "reasoning, in the same turn — the verdict above was reached WITH this evidence already incorporated."
        )
    elif source == "backfill_reconsidered":
        source_note = (
            "<b>Backfilled and reconsidered:</b> this fold was run afterward as a separate batch job, then its "
            "result was sent back to Fable 5 in a dedicated follow-up call that was forced to revise the verdict "
            "above in light of it (see chaperone.reconsider_with_fold) — not left as a disconnected side-note."
        )
    else:
        source_note = ""
    return (
        f"<div class='executed'>"
        f"<span class='tag'>EXECUTED</span>"
        f"Actually re-folded <b>{esc(chains)}</b> via AlphaFast "
        f"(<span class='metric'>ipTM={esc(result.get('iptm'))} pTM={esc(result.get('ptm'))} "
        f"ranking_score={esc(result.get('ranking_score'))}</span>, {esc(result.get('elapsed_seconds'))}s)"
        f"<div style='font-size:12px;color:var(--grey);margin-top:6px;'>{esc(note)}</div>"
        f"{f'<div style=\"font-size:11px;color:var(--teal);margin-top:4px;\">{source_note}</div>' if source_note else ''}"
        f"{analysis_html}"
        f"{structure_html}"
        f"</div>"
    )


def load_tool_calls(row: dict, log_dir: Path) -> list:
    log_path = log_dir / f"{row['protein_a']}__{row['protein_b']}.json"
    if not log_path.exists():
        return []
    try:
        return json.loads(log_path.read_text()).get("_tool_calls", [])
    except json.JSONDecodeError:
        return []


def gather_executed_results(row: dict, tool_calls: list, executed_by_pair: dict) -> list:
    """A fold result can come from two DIFFERENT mechanisms, worth
    distinguishing explicitly rather than blurring together (reported as
    confusing by a user): (1) the hand-curated backfill
    (fold_runs/followups/_summary.json), reconsidered afterward by
    chaperone.reconsider_with_fold — the verdict above was regenerated by a
    dedicated follow-up call citing this evidence; or (2) the agent calling
    fold_complex live, in the same conversation, during its own original
    reasoning (captured in this candidate's own tool-call ledger) — the
    verdict above was reached directly incorporating this evidence."""
    executed_results = [
        {**r, "_source": "backfill_reconsidered"}
        for r in (executed_by_pair or {}).get(f"{row['protein_a']}/{row['protein_b']}", [])
    ]
    for call in tool_calls:
        if call.get("tool") == "fold_complex" and isinstance(call.get("output"), dict) and "iptm" in call["output"]:
            executed_results.append({**call["output"], "status": "ok", "_source": "live_agent"})
    return executed_results


def render_validation_strategy(strategy: dict) -> str:
    if not strategy:
        return ""
    method = METHOD_LABELS.get(strategy["method"], esc(strategy["method"]))
    variant = strategy.get("method_variant")
    summary_label = f"{method} ({variant})" if variant else method
    tissues = ", ".join(normalize_str_list(strategy.get("target_tissues")))
    cell_types = ", ".join(normalize_str_list(strategy.get("cell_types")))
    return f"""
    <details class="validation-strategy" open>
      <summary>Validation strategy: {esc(summary_label)}</summary>
      <div class="field"><div class="k">Why this method</div><div class="v">{esc(strategy.get('method_rationale'))}</div></div>
      <div class="field"><div class="k">Target tissue(s)</div><div class="v">{esc(tissues)}</div></div>
      <div class="field"><div class="k">Why these tissues (HPA)</div><div class="v">{esc(strategy.get('tissue_rationale'))}</div></div>
      {f"<div class='field'><div class='k'>Cell type(s)</div><div class='v'>{esc(cell_types)}</div></div>" if cell_types else ""}
      <div class="field"><div class="k">Protocol notes</div><div class="v">{esc(strategy.get('protocol_notes'))}</div></div>
      <div class="field"><div class="k">Controls</div><div class="v">{esc(strategy.get('controls'))}</div></div>
      <div class="field"><div class="k">Caveats</div><div class="v">{esc(strategy.get('caveats'))}</div></div>
    </details>
    """


def render_candidate(row, tool_calls: list, executed_results: list, structures: dict, strategies: dict = None):
    protein_a, protein_b = row["protein_a"], row["protein_b"]
    pair_id = f"{protein_a}__{protein_b}"

    verdict = row.get("verdict") or "INSUFFICIENT_EVIDENCE"
    color = VERDICT_COLORS.get(verdict, "#999")
    validation = classify_validation(row, executed_results)
    strategy = (strategies or {}).get(f"{protein_a}/{protein_b}")
    strategy_html = render_validation_strategy(strategy)

    evidence_html = ""
    for key, label in EVIDENCE_FIELDS:
        val = row.get(key)
        if val:
            evidence_html += f"<div class='field'><div class='k'>{esc(label)}</div><div class='v'>{esc(val)}</div></div>"

    rationale = row.get("rationale") or ""
    other_subunits = row.get("other_subunits") or ""
    follow_up = row.get("follow_up") or ""

    executed_html = "".join(
        render_executed_follow_up(r, f"{pair_id}_{i}", structures) for i, r in enumerate(executed_results)
    )
    followup_html = executed_html if executed_html else (
        f"<div class='followup'><b>Follow-up (not yet executed):</b> {esc(follow_up)}</div>" if follow_up else ""
    )
    validation_badge = (
        f"<span class='badge' style='background:{validation['color']};opacity:.85;' title='{esc(validation['note'] or validation['reason'])}'>VALIDATE</span>"
        if validation else ""
    )
    live_fold = any(r.get("_source") == "live_agent" for r in executed_results)
    reconsidered_fold = any(r.get("_source") == "backfill_reconsidered" for r in executed_results)
    if live_fold:
        fold_badge = "<span class='badge' style='background:#1B2A4A;' title='The agent called fold_complex live, in this candidate's own conversation, and the verdict reflects that real fold.'>🧬 FOLD-CHECKED (live)</span>"
    elif reconsidered_fold:
        fold_badge = "<span class='badge' style='background:#1B2A4A;' title='A real fold was run afterward and this verdict was regenerated in a dedicated follow-up call citing that result — see the EXECUTED block below.'>🧬 FOLD-CHECKED (reconsidered)</span>"
    else:
        fold_badge = ""

    return f"""
    <div class="card" data-log="{esc(pair_id)}">
      <div class="card-head">
        <span class="pair">{esc(protein_a)} &harr; {esc(protein_b)}</span>
        <span class="badge" style="background:{color}">{esc(verdict)}</span>
        {fold_badge}
        {validation_badge}
        <span class="conf">{esc(row.get('confidence'))} confidence</span>
        <span class="metrics">ipTM {esc(row.get('iptm'))} &middot; pTM {esc(row.get('ptm'))}</span>
      </div>
      <div class="card-body">
        <div class="field"><div class="k">Rationale</div><div class="v">{esc(rationale)}</div></div>
        {evidence_html}
        {"<div class='field'><div class='k'>Other subunits</div><div class='v'>" + esc(other_subunits) + "</div></div>" if other_subunits else ""}
        {followup_html}
        {strategy_html}
        <details class="tools">
          <summary>Full tool-call ledger ({len(tool_calls)} calls) — raw evidence behind this verdict</summary>
          {render_tool_calls(tool_calls)}
        </details>
      </div>
    </div>
    """


def build_report(csv_path: str, log_dir: str, out_path: str, fold_summary_path: str = None):
    rows = list(csv.DictReader(open(csv_path)))
    log_dir = Path(log_dir)
    executed_by_pair = load_executed_follow_ups(
        Path(fold_summary_path) if fold_summary_path
        else PROJECT_ROOT / "fold_runs" / "followups" / "_summary.json"
    )

    counts = {}
    for r in rows:
        counts[r.get("verdict", "?")] = counts.get(r.get("verdict", "?"), 0) + 1
    stats_html = "".join(
        f"<div class='stat'><div class='n'>{n}</div><div class='l'>{esc(v)}</div></div>"
        for v, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )

    structures = {}
    per_row = []
    for row in rows:
        tool_calls = load_tool_calls(row, log_dir)
        executed_results = gather_executed_results(row, tool_calls, executed_by_pair)
        per_row.append((row, tool_calls, executed_results))

    # Clear, explicit breakdown of HOW each fold-checked verdict got that way
    # — "was a fold used" was reported as unclear when this was one blended
    # count, so split it into the two actually-different mechanisms.
    n_live = sum(1 for _, _, ex in per_row if any(r.get("_source") == "live_agent" for r in ex))
    n_reconsidered = sum(1 for _, _, ex in per_row if any(r.get("_source") == "backfill_reconsidered" for r in ex))
    if n_live:
        stats_html += f"<div class='stat' style='background:#EAFBF8;'><div class='n'>{n_live}</div><div class='l'>Fold-checked live, in-conversation</div></div>"
    if n_reconsidered:
        stats_html += f"<div class='stat' style='background:#EAFBF8;'><div class='n'>{n_reconsidered}</div><div class='l'>Fold-checked, verdict reconsidered after</div></div>"
    n_no_fold = len(per_row) - n_live - n_reconsidered
    stats_html += f"<div class='stat' style='background:var(--light);'><div class='n'>{n_no_fold}</div><div class='l'>No fold run (evidence-only verdict)</div></div>"

    strategies = load_validation_strategies(PROJECT_ROOT / "data" / "validation_strategies.json")
    validation_html = render_validation_summary([(row, ex) for row, _, ex in per_row], strategies)
    cards_html = "".join(
        render_candidate(row, tool_calls, executed_results, structures, strategies)
        for row, tool_calls, executed_results in per_row
    )
    js = JS_TEMPLATE.replace("__STRUCTURES_JSON__", json.dumps(structures))

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Deorphanisation triage report</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>{CSS}</style></head>
<body>
<header>
  <h1>Deorphanisation — PPI triage report</h1>
  <p>{len(rows)} candidates &middot; every verdict below is traceable to the raw tool evidence it cites &mdash; click a row to expand{f" &middot; {len(structures)} with a real folded structure to view" if structures else ""}</p>
</header>
<div class="summary">{stats_html}</div>
{validation_html}
<main>{cards_html}</main>
<script>{js}</script>
</body></html>"""

    Path(out_path).write_text(html_doc)
    print(f"Wrote {out_path} ({len(rows)} candidates)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("log_dir")
    ap.add_argument("--out", default="data/report.html")
    ap.add_argument("--fold-summary", default=None, help="fold_runs/followups/_summary.json (auto-detected by default)")
    args = ap.parse_args()
    build_report(args.csv_path, args.log_dir, args.out, args.fold_summary)
