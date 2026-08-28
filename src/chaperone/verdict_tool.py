from typing import List, Literal, Optional

from anthropic import beta_async_tool

VERDICTS = [
    "CONFIRMED_NOVEL",
    "LIKELY_SUBCOMPLEX",
    "ALREADY_KNOWN",
    "IMPLAUSIBLE",
    "LIKELY_ARTIFACT_PTM",
    "INSUFFICIENT_EVIDENCE",
]


def make_record_verdict_tool(sink: dict):
    """Build a record_verdict @beta_tool that writes its input into `sink`
    (a dict passed by the caller) instead of a module-level global, so a
    fresh tool instance is used per candidate in concurrent runs."""

    @beta_async_tool
    async def record_verdict(
        verdict: Literal[
            "CONFIRMED_NOVEL",
            "LIKELY_SUBCOMPLEX",
            "ALREADY_KNOWN",
            "IMPLAUSIBLE",
            "LIKELY_ARTIFACT_PTM",
            "INSUFFICIENT_EVIDENCE",
        ],
        confidence: Literal["high", "medium", "low"],
        rationale: str,
        coexpression_evidence: Optional[str] = None,
        subcellular_evidence: Optional[str] = None,
        ptm_glycosylation_evidence: Optional[str] = None,
        cooccurrence_evidence: Optional[str] = None,
        known_interaction_evidence: Optional[str] = None,
        other_subunits: Optional[List[str]] = None,
        follow_up: Optional[str] = None,
    ) -> str:
        """Record the final verdict for this PPI candidate. Call this
        exactly once, as your last action, after you've pulled the HPA and
        PubMed evidence you need.

        Args:
            verdict: overall classification of the candidate PPI. Use
                LIKELY_ARTIFACT_PTM when expression/localization support the
                pair but literature indicates one or both proteins are
                normally glycosylated or carry PTMs that AF3's bare-sequence
                model wouldn't represent, and that plausibly affects this
                interaction (not just any PTM anywhere on the protein).
            confidence: confidence in this verdict given the evidence pulled.
            rationale: 1-3 sentences citing the actual evidence pulled from tools.
            coexpression_evidence: what tissue/cell-type expression data showed.
            subcellular_evidence: what subcellular location data showed.
            ptm_glycosylation_evidence: what pubmed_ptm_glycosylation showed
                for either protein, and whether it's plausibly relevant to
                the modeled interface (e.g. extracellular domain of a
                membrane/secreted protein) or just an unrelated PTM elsewhere.
            cooccurrence_evidence: what pubmed_cooccurrence showed for this
                gene pair (a nonzero hit count doesn't confirm a known direct
                PPI by itself — note in rationale if it looks unrelated).
            known_interaction_evidence: what string_known_interaction,
                cellphonedb_known_interaction, and pubmed_cooccurrence
                showed. A CellPhoneDB hit is strong direct evidence on its
                own (→ ALREADY_KNOWN). A STRING hit with high "database"/
                "experiments" score is ONLY strong evidence when
                pubmed_cooccurrence also finds at least one real hit for
                this pair (the STRING "database" channel can reflect
                curated pathway/interactome membership rather than a
                published paper on this specific pair) — a strong STRING
                channel with zero PubMed hits for this pair is still novel,
                not ALREADY_KNOWN. A STRING hit driven only by "textmining"
                is the same weak co-mention caveat as PubMed, not
                confirmation, regardless of pubmed_cooccurrence.
            other_subunits: gene symbols of other known complex members this
                pair should be re-modeled with, if any — from
                cellphonedb_complex_members when it covers this gene
                (surface/adhesion complexes), else HPA's weaker
                known_interactions_count proxy (CORUM isn't wired up yet).
            follow_up: suggested next step, e.g. re-run AF3 multimer with an
                extra subunit, re-run with the modified residue represented,
                or check the specific PTM/glycosylation site location once a
                per-residue tool exists.
        """
        sink["verdict"] = {
            "verdict": verdict,
            "confidence": confidence,
            "rationale": rationale,
            "coexpression_evidence": coexpression_evidence,
            "subcellular_evidence": subcellular_evidence,
            "ptm_glycosylation_evidence": ptm_glycosylation_evidence,
            "cooccurrence_evidence": cooccurrence_evidence,
            "known_interaction_evidence": known_interaction_evidence,
            "other_subunits": other_subunits or [],
            "follow_up": follow_up,
        }
        return "verdict recorded"

    return record_verdict
