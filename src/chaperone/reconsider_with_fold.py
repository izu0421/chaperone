"""Feed an EXECUTED follow-up fold's real structural evidence back to Fable 5
so it actually reconsiders the verdict, instead of leaving the fold result
sitting in the report as disconnected evidence next to an unrevised verdict.

Gap this closes: the 75-candidate triage and the follow-up-fold backfill
(chaperone.run_follow_up_folds) were separate steps. The backfill's real
findings (does the interface hold up? PTM/topology conflicts?) got rendered
in the report's "EXECUTED" block, but nothing re-ran the reasoning over that
new evidence — unlike the one live in-agent fold_complex case (ITGA8/F11R),
where the same agent turn immediately incorporated the finding into its
verdict. This script does that reconsideration pass for every backfilled
candidate.

Usage:
    python -m chaperone.reconsider_with_fold
"""
import asyncio
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from anthropic import AsyncAnthropic  # noqa: E402
from .verdict_tool import make_record_verdict_tool  # noqa: E402
from .paths import PROJECT_ROOT as ROOT  # noqa: E402

MODEL = "claude-fable-5"

SYSTEM_PROMPT = """\
You previously (in an earlier, separate step) triaged a candidate \
protein-protein interaction (PPI) predicted by AlphaFold3, and produced a \
verdict based on HPA/PubMed/STRING/CellPhoneDB evidence. Since then, a real \
AlphaFold3 fold of the proposed extended complex was actually run (via \
AlphaFast on real GPUs) — you now have its real ipTM/pTM/ranking_score plus \
structural analysis: interface residues, per-interface pLDDT, whether a \
known UniProt PTM/glycosylation site sits ON the modeled interface, and \
which side of the membrane (Cytoplasmic/Extracellular/Transmembrane) the \
interface falls on.

Your job: reconsider the original verdict in light of this real evidence, \
and record an updated verdict — do not just repeat the old one unless the \
new evidence genuinely doesn't change anything.

Guidance on the new evidence:
- If the extended-complex fold's interface ipTM/pLDDT is high and consistent \
with the original hypothesis, that's real support — consider whether \
CONFIRMED_NOVEL or LIKELY_SUBCOMPLEX (with higher confidence) is now \
warranted.
- If adding the proposed subunit did NOT improve the interface (low ipTM at \
the specific chain-pair interface, low interface pLDDT), the missing-subunit \
hypothesis is not supported — say so plainly, and don't just default back to \
the original verdict without explaining why the fold didn't help.
- A known PTM/glycosylation site landing directly ON the modeled interface \
is real, specific evidence for LIKELY_ARTIFACT_PTM — much stronger than the \
original protein-level literature guess.
- An interface touching BOTH Cytoplasmic AND Extracellular residues at once \
is worth noting (a real membrane can't be crossed like that) but is NOT \
automatic proof by itself — this can happen even at high pLDDT because AF3 \
never models a lipid bilayer, so don't over-weight it alone; weigh it \
alongside the other evidence.
- If the fold job errored/failed, say so and fall back to the original \
evidence — you have no new information in that case.

You MUST end by calling record_verdict exactly once, with a rationale that \
explicitly engages with what the real fold showed (not just restating the \
original rationale)."""


def format_structural_analysis(sa: dict) -> str:
    if not sa or not sa.get("interfaces"):
        return "(no structural analysis available)"
    lines = []
    for iface in sa["interfaces"]:
        ptm_a = iface.get("ptm_sites_at_interface_a") or []
        ptm_b = iface.get("ptm_sites_at_interface_b") or []
        topo_a = iface.get("interface_topology_a") or []
        topo_b = iface.get("interface_topology_b") or []
        pa = (iface.get("interface_plddt_a") or {}).get("mean")
        pb = (iface.get("interface_plddt_b") or {}).get("mean")
        line = (
            f"- Interface {iface['chain_a']}-{iface['chain_b']}: "
            f"{iface.get('n_interface_residues_a')}/{iface.get('n_interface_residues_b')} residues, "
            f"pLDDT {pa}/{pb}"
        )
        if ptm_a or ptm_b:
            descs = [p["description"] for p in [*ptm_a, *ptm_b]]
            line += f"; PTM/glycosylation AT interface: {'; '.join(descs)}"
        both_sides = "Cytoplasmic" in (set(topo_a) | set(topo_b)) and "Extracellular" in (set(topo_a) | set(topo_b))
        if topo_a or topo_b:
            line += f"; membrane topology at interface: {sorted(set(topo_a) | set(topo_b))}"
            if both_sides:
                line += " (touches both sides of the membrane at once)"
        lines.append(line)
    return "\n".join(lines)


async def reconsider_one(client: AsyncAnthropic, row: dict, fold_result: dict) -> dict:
    sink: dict = {}
    record_verdict = make_record_verdict_tool(sink)
    tool_schema = record_verdict.to_dict()

    if fold_result.get("status") != "ok":
        user_prompt = (
            f"Candidate: {row['protein_a']} <-> {row['protein_b']}\n"
            f"Original verdict: {row['verdict']} ({row['confidence']} confidence)\n"
            f"Original rationale: {row['rationale']}\n\n"
            f"The proposed follow-up fold FAILED: {fold_result.get('error')}\n"
            "No new evidence — re-record the original verdict as-is, noting the failed attempt in follow_up."
        )
    else:
        chains = " + ".join(c["identifier"] for c in fold_result.get("chains", []))
        user_prompt = (
            f"Candidate: {row['protein_a']} <-> {row['protein_b']}\n"
            f"Original verdict: {row['verdict']} ({row['confidence']} confidence)\n"
            f"Original rationale: {row['rationale']}\n"
            f"Original other_subunits: {row['other_subunits']}\n"
            f"Original follow_up: {row['follow_up']}\n\n"
            f"REAL FOLD EXECUTED: {chains}\n"
            f"ipTM={fold_result.get('iptm')} pTM={fold_result.get('ptm')} "
            f"ranking_score={fold_result.get('ranking_score')}\n"
            f"Structural analysis:\n{format_structural_analysis(fold_result.get('structural_analysis'))}\n\n"
            "Reconsider the verdict given this real evidence and record it."
        )

    response = await client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "record_verdict"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_verdict":
            return {**row, **block.input}

    raise RuntimeError(f"No record_verdict tool call in response for {row['protein_a']}/{row['protein_b']}")


async def run_fold_reconsideration(csv_path: Path, fold_summary_path: Path, concurrency: int = 4) -> int:
    """Runs reconsideration over every candidate with a REAL executed
    follow-up fold in fold_summary_path, and rewrites csv_path in place.
    Returns the number of candidates reconsidered. A no-op (returns 0) if no
    follow-up folds have been executed yet — this only revisits candidates
    where real structural evidence actually exists to reconsider against."""
    if not fold_summary_path.exists():
        return 0

    client = AsyncAnthropic()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    fold_results = {r["source_pair"]: r for r in json.loads(fold_summary_path.read_text())}

    row_by_pair = {f"{r['protein_a']}/{r['protein_b']}": r for r in rows}
    targets = [(row_by_pair[pair], fold) for pair, fold in fold_results.items() if pair in row_by_pair]
    if not targets:
        return 0
    print(f"Reconsidering {len(targets)} candidates with executed folds...")

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(row, fold):
        async with semaphore:
            try:
                updated = await reconsider_one(client, row, fold)
                print(f"  {row['protein_a']}/{row['protein_b']}: {row['verdict']} -> {updated['verdict']}")
                return updated
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {row['protein_a']}/{row['protein_b']}: {exc}")
                return row

    updated_rows = await asyncio.gather(*(worker(row, fold) for row, fold in targets))
    updated_by_pair = {f"{r['protein_a']}/{r['protein_b']}": r for r in updated_rows}

    final_rows = [updated_by_pair.get(f"{r['protein_a']}/{r['protein_b']}", r) for r in rows]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in final_rows:
            out = {k: r.get(k) for k in fieldnames}
            if isinstance(out.get("other_subunits"), list):
                out["other_subunits"] = ";".join(out["other_subunits"])
            writer.writerow(out)

    print(f"Updated {csv_path}")
    return len(targets)


async def main():
    load_dotenv(ROOT / ".env")
    csv_path = ROOT / "data" / "verdicts_full.csv"
    summary_path = ROOT / "fold_runs" / "followups" / "_summary.json"
    await run_fold_reconsideration(csv_path, summary_path)


if __name__ == "__main__":
    asyncio.run(main())
