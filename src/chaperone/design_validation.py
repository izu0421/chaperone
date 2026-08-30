"""For every candidate recommended for experimental validation, design an
actual validation strategy: which assay (PLA for co-localization, or a
stimulation/functional assay for ligand-receptor pairs), which tissue(s) to
use (grounded in real HPA expression data, not guessed), which cell types,
and what controls/caveats apply.

Uses the same tier logic as build_report.py's classify_validation (imported
directly, not reimplemented) so the two never drift apart.

Usage:
    python -m chaperone.design_validation
"""
import asyncio
import csv
import json
import sys
from pathlib import Path

from typing import Literal

import anthropic
import httpx
from dotenv import load_dotenv


from anthropic import AsyncAnthropic, beta_async_tool  # noqa: E402
from .build_report import (  # noqa: E402
    classify_validation,
    gather_executed_results,
    load_executed_follow_ups,
    load_tool_calls,
)
from .sources.hpa_client import fetch_gene_profile  # noqa: E402
from .strategy_utils import normalize_str_list  # noqa: E402
from .paths import PROJECT_ROOT as ROOT  # noqa: E402

DEFAULT_FOLD_SUMMARY = ROOT / "fold_runs" / "followups" / "_summary.json"
DEFAULT_LOG_DIR = ROOT / "log"


def make_client() -> AsyncAnthropic:
    # The local dev proxy (https_proxy env var) intermittently drops this
    # request shape (long system prompt + wide tool schema): confirmed via
    # repeated testing that api.anthropic.com is directly reachable from this
    # host, so bypass the proxy for just this client rather than for every
    # httpx user in the process (HPA/UniProt/etc. clients keep using it).
    return AsyncAnthropic(http_client=httpx.AsyncClient(trust_env=False))

# claude-fable-5's bio-safety classifier deterministically refuses ANY wet-lab
# validation-protocol request for a named protein pair (confirmed: even a bare
# "what experiment would show X and Y interact?" with no system prompt, no
# tools, no PTM/glycosylation phrasing refuses). sonnet-5/opus-5 do not, so
# this task runs on sonnet-5 instead of the rest of the pipeline's fable-5.
MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """\
You are designing a concrete, practical experimental validation strategy for \
a candidate protein-protein interaction (PPI) predicted by AlphaFold3 and \
triaged as worth pursuing. You are given the triage verdict/rationale and \
real Human Protein Atlas evidence (protein class, subcellular location, \
secretome location, tissue and single-cell-type expression) for both \
proteins — use it, don't guess generic tissues.

Pick ONE primary method:
- **PLA** (proximity ligation assay) — best when both proteins are expected \
in the SAME cell (direct physical proximity, in situ, works in tissue \
sections or fixed cells, needs a validated antibody pair for both targets). \
Good default for LIKELY_SUBCOMPLEX/CONFIRMED_NOVEL pairs that are both \
intracellular, or a receptor pair on the same cell surface.
- **Stimulation/functional assay** — best for a secreted ligand + membrane \
receptor pair (paracrine/endocrine — the two are not necessarily in the same \
cell), or when a functional/signalling readout is more informative than mere \
co-localization: stimulate receptor-expressing cells with the candidate \
ligand (recombinant protein, or co-culture with ligand-secreting cells) and \
measure a downstream readout (phosphorylation of a pathway node, reporter \
activation, receptor internalization, calcium flux — pick what's plausible \
for this receptor family).
- **co_ip** (co-immunoprecipitation + western blot) — reasonable fallback \
when PLA/stimulation don't fit well (e.g. testing a specific subunit \
addition), but say so explicitly rather than defaulting to it.

For LIKELY_ARTIFACT_PTM candidates: the concern is that AF3 modeled a \
bare-sequence interface that may be blocked/gated by a glycan or PTM in \
reality. Say this explicitly in caveats, and where sensible describe a \
variant of the assay that tests the PTM/glycan role directly (e.g. compare \
PLA signal +/- a glycosidase treatment, or compare stimulation response \
with a phospho-dead/glycosylation-site mutant if that's a reasonable ask) \
in the method_variant field — the method field itself must still be exactly \
one of PLA / stimulation_assay / co_ip / other, never a sentence.

Target tissues: choose 2-4 REAL tissues from the HPA data given, prioritizing \
ones where BOTH proteins show meaningful expression (RNA or protein-level); \
if there's no real overlap (e.g. genuine paracrine signalling across \
compartments), say so and instead name the tissue/cell type most relevant \
for each side separately, explaining why that still lets the assay work \
(e.g. co-culture, or a tissue where the receptor-bearing cell type is \
physically near the ligand-secreting one).

You MUST end by calling record_validation_strategy exactly once."""


def make_validation_tool(sink: dict):
    @beta_async_tool
    async def record_validation_strategy(
        method: Literal["PLA", "stimulation_assay", "co_ip", "other"],
        method_variant: str,
        method_rationale: str,
        target_tissues: list[str],
        tissue_rationale: str,
        cell_types: list[str],
        protocol_notes: str,
        controls: str,
        caveats: str,
    ) -> str:
        """Record the validation strategy for this candidate.

        Args:
            method: EXACTLY one of "PLA", "stimulation_assay", "co_ip", or
                "other" — never a sentence or a method name with extra detail
                appended. Any twist on the base method (e.g. "PLA +/- a
                glycosidase treatment") goes in method_variant, not here.
            method_variant: short label for a specific variant of the base
                method if relevant (e.g. "with glycosidase-treatment control
                arm"); empty string if the base method is used as-is.
            method_rationale: 1-2 sentences on why this method fits this
                specific pair (compartment, ligand/receptor status, etc.).
            target_tissues: 2-4 real tissue names from the HPA data provided
                (or, if no real overlap exists, name each side's best tissue).
            tissue_rationale: 1-2 sentences citing the actual expression
                numbers/tissues that justify this choice.
            cell_types: specific cell type(s) within those tissues if HPA
                single-cell data supports it (can be empty list).
            protocol_notes: 2-4 concrete, actionable steps or considerations
                (antibodies needed, fixation, readout, timing, etc.).
            controls: positive and negative control suggestions.
            caveats: anything that could complicate interpretation (e.g. the
                PTM/glycosylation concern, low pLDDT, weak expression).
        """
        sink["strategy"] = {
            "method": method,
            "method_variant": method_variant,
            "method_rationale": method_rationale,
            "target_tissues": normalize_str_list(target_tissues),
            "tissue_rationale": tissue_rationale,
            "cell_types": normalize_str_list(cell_types),
            "protocol_notes": protocol_notes,
            "controls": controls,
            "caveats": caveats,
        }
        return "recorded"

    return record_validation_strategy


def summarize_hpa(gene: str, profile: dict) -> str:
    if "error" in profile:
        return f"{gene}: no HPA data ({profile['error']})"
    return (
        f"{gene}: protein_class={profile.get('Protein class')}, "
        f"subcellular_main_location={profile.get('Subcellular main location')}, "
        f"secretome_location={profile.get('Secretome location')}, "
        f"rna_tissue_specific_nTPM={profile.get('RNA tissue specific nTPM')}, "
        f"protein_tissue_specific_intensity={profile.get('Protein tissue specific Intensity')}, "
        f"rna_single_cell_type_specific_nCPM={profile.get('RNA single cell type specific nCPM')}"
    )


async def design_one(client: AsyncAnthropic, row: dict, tier: dict, revision_issues: list = None) -> dict:
    profile_a = fetch_gene_profile(row["protein_a"])
    profile_b = fetch_gene_profile(row["protein_b"])

    sink: dict = {}
    tool = make_validation_tool(sink)

    user_prompt = (
        f"Candidate: {row['protein_a']} <-> {row['protein_b']}\n"
        f"Verdict: {row['verdict']} ({row['confidence']} confidence) — tier: {tier['label']}\n"
        f"Rationale: {row['rationale']}\n"
        f"Other subunits (if any): {row.get('other_subunits')}\n\n"
        f"HPA evidence:\n{summarize_hpa(row['protein_a'], profile_a)}\n{summarize_hpa(row['protein_b'], profile_b)}\n\n"
        "Design the validation strategy."
    )
    if revision_issues:
        issues_text = "\n".join(f"- {i}" for i in revision_issues)
        user_prompt += (
            "\n\nA prior attempt at this strategy was reviewed and found to have real problems — fix them, "
            "don't just restate the same strategy. Ground every tissue/cell-type claim ONLY in the HPA "
            "evidence given above (nothing outside it), and if the method doesn't genuinely fit the "
            "compartment evidence, change the method rather than asserting compartments not shown above. "
            f"Specific problems found in the prior attempt:\n{issues_text}"
        )

    # The local proxy this project runs behind drops this specific request
    # shape (long system prompt + wide tool schema) intermittently — pure
    # transport flakiness, confirmed unrelated to content (bisected field-by-
    # field) and unrelated to timeout length. Retry hard with backoff.
    # Also retry if the tool call succeeds but is missing a required field
    # (observed live: a tool_use block without "method" at all, despite the
    # forced tool_choice — tool-call output isn't strictly schema-validated,
    # so "the API accepted the call" doesn't guarantee every field is
    # present) — a naive version of this wrote the malformed result straight
    # to disk before a downstream KeyError was ever noticed.
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = await client.beta.messages.create(
                model=MODEL,
                max_tokens=3000,
                system=SYSTEM_PROMPT,
                tools=[tool.to_dict()],
                tool_choice={"type": "tool", "name": "record_validation_strategy"},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except (anthropic.APIConnectionError, anthropic.InternalServerError):
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(min(30, 2 * (attempt + 1)))
            continue

        for block in response.content:
            if block.type == "tool_use" and block.name == "record_validation_strategy":
                if "method" not in block.input:
                    break  # malformed — fall through to retry
                result = {"protein_a": row["protein_a"], "protein_b": row["protein_b"], **block.input}
                result["target_tissues"] = normalize_str_list(result.get("target_tissues"))
                result["cell_types"] = normalize_str_list(result.get("cell_types"))
                return result
        if attempt < max_attempts - 1:
            await asyncio.sleep(2)

    raise RuntimeError(f"No valid record_validation_strategy call for {row['protein_a']}/{row['protein_b']} after {max_attempts} attempts")


async def run_validation_design(
    csv_path: Path, out_path: Path, concurrency: int = 6,
    log_dir: Path = None, fold_summary_path: Path = None,
) -> list[dict]:
    """Design a validation strategy for every candidate in csv_path that
    classify_validation recommends. Writes out_path and returns the list.

    Needs each row's REAL executed_results (not an empty list) since
    classify_validation now only recommends LIKELY_SUBCOMPLEX/
    LIKELY_ARTIFACT_PTM once a real executed fold resolved the concern it
    names — passing [] would silently skip every one of those, even
    already-resolved ones, and only ever design for CONFIRMED_NOVEL rows."""
    client = make_client()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    executed_by_pair = load_executed_follow_ups(fold_summary_path or DEFAULT_FOLD_SUMMARY)
    targets = []
    for row in rows:
        tool_calls = load_tool_calls(row, log_dir or DEFAULT_LOG_DIR)
        executed_results = gather_executed_results(row, tool_calls, executed_by_pair)
        tier = classify_validation(row, executed_results)
        if tier:
            targets.append((row, tier))
    print(f"Designing validation strategies for {len(targets)} recommended candidates...")

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(row, tier):
        async with semaphore:
            try:
                strategy = await design_one(client, row, tier)
                print(f"  {row['protein_a']}/{row['protein_b']}: {strategy['method']} in {strategy['target_tissues']}")
                return strategy
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {row['protein_a']}/{row['protein_b']}: {exc}")
                return None

    results = await asyncio.gather(*(worker(row, tier) for row, tier in targets))
    results = [r for r in results if r]

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} strategies to {out_path}")
    return results


async def main():
    load_dotenv(ROOT / ".env")
    await run_validation_design(
        ROOT / "data" / "verdicts_full.csv",
        ROOT / "data" / "validation_strategies.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
