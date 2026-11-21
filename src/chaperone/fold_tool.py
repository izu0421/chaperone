"""Gives the triage agent itself the ability to actually run a fold, not just
recommend one. Wraps chaperone.fold_candidate.fold() as a tool the triage
agent can call mid-reasoning, so a "this looks like it needs subunit X"
hypothesis gets tested immediately with a real AF3 result — the finding goes
into the verdict as evidence, not into `follow_up` as a deferred suggestion.

Guardrails (folding is real GPU compute, ~9 min cold per job, on a shared
box with only 2-3 GPUs):
- capped at one fold per candidate (a second attempt gets a clear refusal,
  not a queued second job)
- bounded by a semaphore shared across the whole batch run, sized to the
  number of GPUs actually made available to this run
"""
import asyncio
import itertools
import json
import re
import time

from anthropic import beta_async_tool

from .analyze_fold import analyze_fold_result, summarize
from .fold_candidate import fold as fold_sync

_gpu_cycle_lock = asyncio.Lock()


def make_fold_tool(pair_id: str, fold_semaphore: asyncio.Semaphore, gpu_counter: itertools.cycle,
                    tool_call_log: list = None, shared_state: dict = None):
    """Build a fold_complex @beta_async_tool scoped to one candidate.

    gpu_counter: a shared itertools.cycle(gpu_devices) so concurrent fold
    calls across candidates round-robin GPUs instead of colliding on one.
    tool_call_log: same list LoggingSession appends to for MCP tools — this
    local tool isn't MCP-routed so LoggingSession never sees it, but the
    report/traceability code needs a record of it too, so we append here
    ourselves in the same {tool, input, output, is_error, elapsed_seconds}
    shape.
    shared_state: pass a dict in (instead of leaving this to create its own
    local one) to let a sibling tool — resolve_ptm_artifact_tool.py — see
    whether fold_complex ran and reuse its FULL structural analysis (this
    tool only returns/logs the token-budget-friendly summarize()'d version
    to the LLM; resolve_ptm_artifact needs the full interface_residues_a/b
    lists to check whether a re-fold's interface still touches the
    originally-flagged residues, so the unsummarized version is stashed
    here rather than recomputed from scratch).
    """
    state = shared_state if shared_state is not None else {}
    state["used"] = False

    @beta_async_tool
    async def fold_complex(proteins: list[str], reason: str,
                           ligand_smiles: str = None, ligand_ccd: str = None,
                           ligand_name: str = None) -> dict:
        """Actually fold a proposed protein complex with AlphaFast (real
        AlphaFold 3, running on this box's GPUs) and return real ipTM/pTM/
        ranking_score — not a guess. Use this when you've proposed a
        concrete, testable structural hypothesis your OTHER tools can't
        settle: e.g. "this pair is missing an obligate subunit" (include the
        original pair + the proposed additional subunit(s) as the protein
        list) or "should really be modeled against its known partner Y
        instead" (include Y in place of/alongside the original partner).

        Also use it when one protein is strictly intracellular and the other
        is a membrane protein — the result's `interface_topology_a/b` (real
        UniProt Cytoplasmic/Extracellular/Transmembrane annotations) tells
        you which side of the membrane the modeled contact actually falls
        on, rather than you having to guess. Cytoplasmic-side is plausible;
        extracellular/luminal-side against a strictly intracellular partner
        is not (an intact membrane is a hard barrier). An interface touching
        BOTH Cytoplasmic AND Extracellular residues at once (seen for real
        on an artifact case in this dataset) is worth noting, but not
        automatic proof either way — this can show up even at high interface
        pLDDT (checked empirically), because AF3 never models a lipid
        bilayer, so a confident fold can still place topologically opposite
        regions spatially close with nothing keeping them apart.

        The fold itself still can't represent glycans/PTMs (bare-sequence
        AF3), but the result includes real structural analysis: interface
        residues, per-interface pLDDT, and whether any known UniProt PTM/
        glycosylation site (an exact residue number, not just "this protein
        has PTMs somewhere") falls ON the modeled interface — a much
        stronger, residue-level signal for a LIKELY_ARTIFACT_PTM call than
        protein-level literature alone. A PTM site actually at the interface
        is real evidence; sites elsewhere on the protein are not relevant.

        This is real GPU compute (several minutes), so: only call this when
        the result would actually change your verdict or confidence — not
        for every LIKELY_SUBCOMPLEX candidate reflexively. You get exactly
        one call; a second call in this conversation will be refused.

        For a SMALL-MOLECULE partner, pass the receptor in `proteins` and the
        molecule in `ligand_smiles` (from resolve_small_molecule) or
        `ligand_ccd` — it is folded into the complex as its own chain, which
        is what a metabolite candidate actually is. Read the result with the
        pocket caveat in mind: AF3 places a small molecule in a receptor
        pocket confidently almost regardless of which receptor it is given,
        so a high interface pTM here is NOT evidence of a specific
        interaction the way it is for a protein-protein interface. What the
        fold can still tell you is whether the molecule sits in the pocket at
        all and which residues line it.

        Args:
            proteins: gene symbols (or UniProt accessions) to fold together
                as one complex, 1-6 chains, in the order they should be
                assigned chain IDs A, B, C... A small-molecule candidate
                normally passes the single receptor here.
            reason: one sentence on what hypothesis this tests.
            ligand_smiles: SMILES of a small molecule to co-fold as its own
                chain, e.g. the `smiles` returned by resolve_small_molecule.
            ligand_ccd: PDB chemical-component code of the molecule instead
                of a SMILES string, e.g. "ATP". Use one or the other.
            ligand_name: how to label the molecule in the report.
        """
        call_t0 = time.time()
        ligands = []
        if ligand_ccd or ligand_smiles:
            ligands.append({"ccd": ligand_ccd, "smiles": ligand_smiles,
                            "name": ligand_name or ligand_ccd})
        input_args = {"proteins": proteins, "reason": reason, "ligands": ligands}

        def log(output, is_error=False):
            if tool_call_log is not None:
                tool_call_log.append({
                    "tool": "fold_complex", "input": input_args, "output": output,
                    "is_error": is_error, "elapsed_seconds": round(time.time() - call_t0, 3),
                })

        if state["used"]:
            err = {
                "error": "fold_complex already used once for this candidate — "
                "proceed to record_verdict with the result you already have."
            }
            log(err, is_error=True)
            return json.dumps(err)
        state["used"] = True

        async with fold_semaphore:
            async with _gpu_cycle_lock:
                gpu_device = next(gpu_counter)
            job_name = re.sub(r"[^a-z0-9_]", "", f"{pair_id.lower()}_agentfold")
            t0 = time.time()
            try:
                result = await asyncio.to_thread(
                    fold_sync, proteins, job_name, None, gpu_device, (1,), False, ligands
                )
            except Exception as exc:  # noqa: BLE001
                err = {"error": f"fold failed: {exc}", "elapsed_seconds": round(time.time() - t0, 1)}
                log(err, is_error=True)
                return json.dumps(err)
        result["reason"] = reason
        try:
            full_analysis = analyze_fold_result(result)
            result["structural_analysis"] = summarize(full_analysis)
            state["full_result"] = {**result, "structural_analysis": full_analysis}
        except Exception as exc:  # noqa: BLE001
            result["structural_analysis_error"] = str(exc)
        log(result)
        return json.dumps(result)

    return fold_complex
