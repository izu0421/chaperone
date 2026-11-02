"""Gives the triage agent the ability to actually attempt resolving a
PTM/glycosylation-artifact concern mid-reasoning, instead of writing
LIKELY_ARTIFACT_PTM and leaving it as a permanent caveat. Wraps
chaperone.resolve_ptm_artifact.resolve() (see skills/resolve-ptm-artifact/
SKILL.md for the full decision rubric) as a second tool, sitting right next
to fold_complex (fold_tool.py) and sharing its GPU budget.

Only usable after fold_complex has already run for this candidate — it
reuses that FULL structural analysis (via the shared_state dict fold_tool.py
now stashes it in) as the "bare fold" so this doesn't spend a second GPU job
re-deriving something already computed; it only spends one more job (the
glycan-aware re-fold) when the bare result actually has a glycan-fixable
concern to test.
"""
import asyncio
import itertools
import json
import re
import time

from anthropic import beta_async_tool

from . import fold_tool
from .resolve_ptm_artifact import resolve as resolve_sync


def make_resolve_ptm_artifact_tool(
    pair_id: str, protein_a: str, protein_b: str,
    fold_semaphore: asyncio.Semaphore, gpu_counter: itertools.cycle,
    fold_state: dict, tool_call_log: list = None,
):
    """fold_state must be the SAME dict passed as `shared_state` to
    make_fold_tool for this candidate — that's how this tool finds the
    already-computed bare fold to build on."""
    state = {"used": False}

    @beta_async_tool
    async def resolve_ptm_artifact(reason: str) -> dict:
        """Actually try to resolve a PTM/glycosylation concern flagged by
        fold_complex's structural_analysis, instead of just writing
        LIKELY_ARTIFACT_PTM and stopping. Requires fold_complex to have
        already been called for this candidate — call this right after, when
        its structural_analysis showed a real Glycosylation site sitting on
        the modeled interface.

        What it does: fetches the real UniProt N-glycosylation sites for
        both proteins, and — if the flagged concern is fixable (an N-linked
        glycan; lipidation and other PTM types aren't currently modelable) —
        runs ONE additional real AlphaFold3 fold with the actual glycan
        (the standard core trisaccharide) covalently attached at its real
        site(s), then checks whether the original interface survives contact
        with it, moves to a different site, or whether nothing reaches a
        real confidence bar at all (chain_pair_iptm >= 0.5 — the standard
        AF-Multimer/AF3 threshold for "the model is actually predicting an
        interaction here"; below that, treat it as no prediction, not a weak
        one, however biologically suggestive a residue pattern looks).

        Returns one of five outcomes: RESOLVED_NO_CONCERN (nothing to fix),
        CONFIRMED_ARTIFACT (a topology violation — not glycan-fixable, no
        second fold was run), NOT_AUTOMATICALLY_RESOLVABLE (flagged PTM type
        isn't a modelable glycan), RESOLVED_NOT_ARTIFACT or
        RESOLVED_ALTERNATE_SITE (the glycan-aware re-fold found >=0.5
        confidence somewhere), or INCONCLUSIVE_LOW_CONFIDENCE (it didn't —
        this is a legitimate, real answer, not a failure; report it as such
        and defer to the wet-lab follow_up rather than treating a
        low-confidence residue pattern as a finding).

        Real GPU compute when it does run a second fold (~10-15 min), capped
        at one call per candidate like fold_complex.

        Args:
            reason: one sentence on what concern this is trying to resolve.
        """
        call_t0 = time.time()
        input_args = {"reason": reason}

        def log(output, is_error=False):
            if tool_call_log is not None:
                tool_call_log.append({
                    "tool": "resolve_ptm_artifact", "input": input_args, "output": output,
                    "is_error": is_error, "elapsed_seconds": round(time.time() - call_t0, 3),
                })

        if state["used"]:
            err = {"error": "resolve_ptm_artifact already used once for this candidate."}
            log(err, is_error=True)
            return json.dumps(err)
        state["used"] = True

        bare_result = fold_state.get("full_result")
        if bare_result is None:
            err = {"error": "fold_complex hasn't been called yet (or it errored) — call it "
                             "first, this tool builds on its structural_analysis."}
            log(err, is_error=True)
            return json.dumps(err)

        async with fold_semaphore:
            async with fold_tool._gpu_cycle_lock:
                gpu_device = next(gpu_counter)
            job_name = re.sub(r"[^a-z0-9_]", "", pair_id.lower())
            t0 = time.time()
            try:
                record = await asyncio.to_thread(
                    resolve_sync, protein_a, protein_b, job_name, gpu_device, bare_result
                )
            except Exception as exc:  # noqa: BLE001
                err = {"error": f"resolve_ptm_artifact failed: {exc}",
                       "elapsed_seconds": round(time.time() - t0, 1)}
                log(err, is_error=True)
                return json.dumps(err)

        record["reason"] = reason
        log(record)
        return json.dumps(record, default=str)

    return resolve_ptm_artifact
