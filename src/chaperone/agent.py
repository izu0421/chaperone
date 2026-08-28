"""Per-candidate agent loop: Fable 5 + HPA MCP tools -> structured verdict."""
import asyncio
import sys
import itertools
import json
import time
from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .fold_tool import make_fold_tool
from .verdict_tool import make_record_verdict_tool

MODEL = "claude-fable-5"

# Default fold-capable GPUs for the agent's own fold_complex tool calls.
# Deliberately excludes GPU 2 by default on this box — it's frequently used
# by other users; override via evaluate_candidate's fold_gpu_devices param.
DEFAULT_FOLD_GPUS = ["0", "1"]


class LoggingSession:
    """Wraps a ClientSession so every MCP tool call/result is captured
    verbatim into `log` — this is what makes a verdict traceable back to the
    actual HPA/PubMed/STRING/CellPhoneDB response, not just the model's
    paraphrase of it. async_mcp_tool only ever calls `.call_tool(...)` on the
    session object it's given, so this proxy only needs to implement that."""

    def __init__(self, session: ClientSession, log: list):
        self._session = session
        self._log = log

    async def call_tool(self, name: str, arguments: dict):
        t0 = time.time()
        result = await self._session.call_tool(name=name, arguments=arguments)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        parsed = []
        for t in texts:
            try:
                parsed.append(json.loads(t))
            except (json.JSONDecodeError, TypeError):
                parsed.append(t)
        self._log.append(
            {
                "tool": name,
                "input": arguments,
                "output": parsed if len(parsed) != 1 else parsed[0],
                "is_error": bool(getattr(result, "isError", False)),
                "elapsed_seconds": round(time.time() - t0, 3),
            }
        )
        return result

SYSTEM_PROMPT = """\
You are triaging a candidate protein-protein interaction (PPI) predicted by \
AlphaFold3, using evidence from the Human Protein Atlas (HPA) and PubMed via \
your tools.

You have nine evidence tools: hpa_expression, hpa_subcellular_location, \
hpa_protein_class, pubmed_ptm_glycosylation, pubmed_cooccurrence, \
string_known_interaction, cellphonedb_known_interaction, \
cellphonedb_complex_members, and uniprot_annotation. You do NOT yet have \
CORUM (general complex database) — cellphonedb_complex_members only covers \
cell-surface/adhesion complexes, so a miss there doesn't rule out a larger \
complex for intracellular proteins.

You ALSO have fold_complex: it actually runs a real AlphaFold3 fold (via \
AlphaFast, on this box's own GPUs) on any protein list you give it, and \
returns real ipTM/pTM/ranking_score plus structural analysis (interface \
residues, per-interface pLDDT, and whether a known UniProt PTM/glycosylation \
site sits ON the modeled interface) — not a guess. If your evidence points \
to a concrete, testable structural hypothesis that only a real fold can \
settle, call it instead of just writing the guess into follow_up:
- Most commonly "this pair looks like it's missing an obligate subunit" \
(other_subunits would otherwise just be a suggestion; fold the original pair \
+ the proposed subunit(s) and see what the interface confidence actually \
does).
- Also useful for a LIKELY_ARTIFACT_PTM hypothesis: if pubmed_ptm_glycosylation \
found a specific modification but you don't know whether it's actually at \
the predicted contact surface, fold_complex's structural_analysis will tell \
you whether a known PTM/glycosylation site lands on the modeled interface — \
a site actually there is real evidence for the artifact call; PTM literature \
existing but landing elsewhere on the protein is not.

It is real GPU compute (several minutes) and capped at one call per \
candidate, so use it when the result would actually change your verdict/ \
confidence, not reflexively for every LIKELY_SUBCOMPLEX or LIKELY_ARTIFACT_PTM \
case — e.g. skip it when you're already confident from CellPhoneDB/STRING \
evidence alone, or when the "extra subunit" is really a list of \
mutually-exclusive paralogs (fold at most one representative, not all of \
them at once — that would test a complex that can't exist). When you do use \
it, report the real result (ipTM, interface pLDDT, PTM-at-interface finding) \
in follow_up/other_subunits/ptm_glycosylation_evidence instead of a deferred \
recommendation.

For known-interaction / novelty ("is this ALREADY_KNOWN") always call ALL \
THREE of string_known_interaction, cellphonedb_known_interaction (cellphonedb \
also if either protein looks like a ligand/receptor from hpa_protein_class), \
AND pubmed_cooccurrence — don't skip pubmed_cooccurrence just because STRING \
already looks strong:
- A cellphonedb_known_interaction hit is curated and precise — treat as \
confirmed ALREADY_KNOWN on its own.
- A string_known_interaction hit with high "database" or "experiments" score \
is NOT sufficient on its own for ALREADY_KNOWN — STRING's "database" channel \
can reflect curated PATHWAY/interactome membership (e.g. "these two genes are \
both in the Eph-ephrin signalling pathway" per KEGG/Reactome-style curation) \
rather than a published paper establishing this SPECIFIC pair binds. Only \
treat it as ALREADY_KNOWN if pubmed_cooccurrence ALSO returns at least one \
real hit for this pair — a strong STRING channel WITH a literature trace is \
real confirmed evidence; a strong STRING channel with ZERO PubMed hits for \
this pair is still novel and worth pursuing, not ALREADY_KNOWN.
- A string_known_interaction hit driven mostly by "textmining" (database/ \
experiments low or zero) is only a weak co-mention signal regardless of \
pubmed_cooccurrence — read the evidence/titles before concluding anything; \
genes are often co-mentioned for unrelated reasons (same pathway, same \
disease, independent genomic lesions).
- No hit anywhere is fairly strong evidence of genuine novelty.

Use cellphonedb_complex_members on both proteins for the "other subunits" \
check when the pair looks like a surface/adhesion complex; otherwise fall \
back to HPA's known_interactions_count as only a weak proxy for \
LIKELY_SUBCOMPLEX.

For each candidate pair, call the HPA tools for both proteins, then reason \
about plausibility:
- If both proteins are annotated as intracellular and NOT secreted/membrane, \
a direct interaction needs them to be co-expressed in the same tissue and, \
ideally, the same single-cell type. No overlap at all is a strong negative \
signal.
- If one is "Predicted secreted proteins" and the other is a membrane \
receptor, tissue/cell-type coexpression in the same cell is NOT required \
(paracrine/endocrine signalling) — check compartment plausibility instead \
(e.g. is the "receptor" actually annotated at the plasma membrane?).
- **Strictly intracellular protein + membrane protein** (one has no \
membrane/secreted annotation at all, the other is "Predicted membrane \
proteins"): this pairing is only plausible if the interaction sits on the \
membrane protein's CYTOPLASMIC side (e.g. a kinase, adaptor, or scaffold \
docking on a receptor's cytoplasmic tail — common and real) — it is NOT \
plausible against that protein's extracellular/luminal side, since an \
intact membrane is a hard physical barrier no amount of protein dynamism \
crosses. HPA's protein_class/subcellular tools alone can't tell you which \
side a modeled AF3 interface falls on; if you fold this pair (fold_complex), \
its structural_analysis reports `interface_topology_a/b` from real UniProt \
membrane-topology annotations (Cytoplasmic/Extracellular/Transmembrane) — \
use that to actually check, rather than guessing. Cytoplasmic+Transmembrane \
or Extracellular+Transmembrane together at an interface is normal (real \
interfaces routinely include TM-helix-boundary residues); it's specifically \
an interface touching BOTH Cytoplasmic AND Extracellular residues at once \
that's a real physical impossibility (seen for real on an artifact case in \
this dataset) — note this can show up even at HIGH interface pLDDT (checked \
empirically: it does not reliably correlate with low confidence), because \
AF3 never models a lipid bilayer, so a confident fold can still place \
topologically opposite regions spatially close with nothing keeping them \
apart. Don't treat the pattern alone as automatic proof of an artifact; note \
it and, if you're unsure, say the actual 3D structure should be inspected \
directly rather than deciding from the topology labels alone. Without a fold to check, treat the mismatch as \
a soft negative (lower confidence, lean IMPLAUSIBLE) rather than an \
automatic veto — proteins do moonlight, shuttle, or have context-dependent \
localization (check HPA's "additional location" field before penalizing too \
hard), and outright ALREADY_KNOWN/CONFIRMED_NOVEL evidence elsewhere can \
outweigh a plausible cytoplasmic-side story you simply haven't confirmed.
- Mismatched subcellular compartments with no known shuttling (e.g. one \
strictly nuclear, the other strictly mitochondrial, both otherwise \
intracellular) argue against a direct interaction.
- If HPA data is missing/ambiguous for either protein (an error result, or no \
useful location/expression data), prefer INSUFFICIENT_EVIDENCE over guessing.
- Use LIKELY_SUBCOMPLEX only as a low-confidence hint (e.g. combined with a \
high known_interactions_count suggesting a well-studied hub protein) and say \
in follow_up that a CORUM/STRING check is needed to confirm.

PTM/glycosylation artifact check (AF3 predicts from bare, unmodified sequence \
— it does not model glycans and by default does not model most PTMs): if \
expression/localization otherwise support the pair, call uniprot_annotation \
AND pubmed_ptm_glycosylation for both proteins before finalizing. \
uniprot_annotation is the more reliable signal — it's the protein's own real \
UniProt entry (function description + keywords), not a noisy literature \
search that often just returns "protein X as an activation marker" papers \
with nothing to do with PTMs. Specifically check:
- The FUNCTION text for a description of the actual binding MECHANISM — \
e.g. "Lectin that binds beta-galactoside..." (any galectin) means the real \
interaction is fundamentally glycan-mediated, which AF3's bare-sequence \
fold cannot represent AT ALL regardless of which residues the model happens \
to place in contact. This is a stronger, more specific signal than "is \
there a PTM site sitting on the exact modeled interface residues" — the \
mechanism itself is unmodeled, not just one residue's modification.
- keywords for "PTM"-category entries (Glycoprotein, Phosphoprotein, \
Lipoprotein, etc.) and ptm_topology_feature_counts for a nonzero \
Glycosylation or Lipidation count — these are what AF3 genuinely can't \
represent. A nonzero Disulfide bond or Modified residue (e.g. \
phosphorylation) count is NOT the same kind of concern — AF3 usually \
represents disulfide-forming cysteine proximity fine, and a small \
modification rarely blocks a whole interface; don't treat those the same \
as glycosylation/lipidation.
If either protein's function/keywords point to a glycan-mediated or \
lipid-mediated binding mechanism, or pubmed_ptm_glycosylation confirms \
glycosylation of the relevant domain, or the interaction type is normally \
gated by a specific PTM (e.g. a phosphorylation-dependent binding domain), \
treat this as a real caveat — the modeled interface may not exist in vivo, \
or only exists conditionally. Use verdict LIKELY_ARTIFACT_PTM for this case \
instead of CONFIRMED_NOVEL, and say in follow_up what should be re-checked \
(e.g. re-run AF3 with the modified residue/glycan represented, or verify \
the PTM site isn't at the modeled interface once a per-residue tool \
exists). Don't overreact to any PTM literature/keyword existing somewhere \
on a large well-studied protein that's clearly unrelated to this specific \
interface — judge plausible relevance, don't just pattern-match on presence.

You MUST end by calling record_verdict exactly once, citing the actual tool \
results you saw."""


@asynccontextmanager
async def hpa_mcp_session():
    params = StdioServerParameters(command=sys.executable, args=["-m", "chaperone.sources.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def evaluate_candidate(
    client: AsyncAnthropic,
    mcp_session: ClientSession,
    candidate,
    fold_semaphore: asyncio.Semaphore = None,
    fold_gpu_cycle: itertools.cycle = None,
    enable_fold: bool = True,
) -> dict:
    tools_result = await mcp_session.list_tools()
    tool_calls: list = []
    logging_session = LoggingSession(mcp_session, tool_calls)
    mcp_tools = [async_mcp_tool(t, logging_session) for t in tools_result.tools]

    sink: dict = {}
    record_verdict = make_record_verdict_tool(sink)

    extra_tools = []
    if enable_fold:
        fold_semaphore = fold_semaphore or asyncio.Semaphore(len(DEFAULT_FOLD_GPUS))
        fold_gpu_cycle = fold_gpu_cycle or itertools.cycle(DEFAULT_FOLD_GPUS)
        extra_tools.append(make_fold_tool(candidate.pair_id, fold_semaphore, fold_gpu_cycle, tool_calls))

    extra_lines = "\n".join(
        f"{k}={v}" for k, v in candidate.extra.items() if v not in (None, "") and not k.endswith("_sequence")
    )
    user_prompt = (
        f"Candidate PPI: {candidate.protein_a} <-> {candidate.protein_b}\n"
        f"AF3 confidence: ipTM={candidate.iptm}, pTM={candidate.ptm}, "
        f"PAE_interaction={candidate.pae_interaction}\n"
        + (f"Additional supplied evidence:\n{extra_lines}\n" if extra_lines else "")
        + "\nNote: n_samples_colocalized/prop_samples_colocalized, if given, are "
        "REAL observed spatial co-localization from actual samples (not "
        "AF3/HPA inference) — this is stronger evidence than tissue/cell-type "
        "enrichment heuristics and should anchor your coexpression reasoning "
        "when present; use the HPA tools to add compartment/plausibility "
        "context around it, not to override it.\n\n"
        "Pull HPA/PubMed/STRING/CellPhoneDB evidence for both proteins and "
        "record your verdict."
    )

    messages = [{"role": "user", "content": user_prompt}]
    transcript = []

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        tools=[*mcp_tools, record_verdict, *extra_tools],
        messages=messages,
    )

    async for message in runner:
        transcript.append(message.model_dump(mode="json"))

    if "verdict" not in sink:
        return {
            "pair": [candidate.protein_a, candidate.protein_b],
            "verdict": "INSUFFICIENT_EVIDENCE",
            "confidence": "low",
            "rationale": "Agent did not call record_verdict.",
            "coexpression_evidence": None,
            "subcellular_evidence": None,
            "other_subunits": [],
            "follow_up": "Retry this candidate.",
            "_transcript": transcript,
            "_tool_calls": tool_calls,
        }

    result = {
        "pair": [candidate.protein_a, candidate.protein_b],
        **sink["verdict"],
        "_transcript": transcript,
        "_tool_calls": tool_calls,
    }
    return result


async def evaluate_candidate_standalone(client: AsyncAnthropic, candidate) -> dict:
    """Convenience wrapper that opens its own MCP session — use for one-off
    calls; for a batch, reuse a single hpa_mcp_session() across candidates
    (see chaperone.run_pipeline) so the HPA server process isn't restarted
    per candidate."""
    async with hpa_mcp_session() as session:
        return await evaluate_candidate(client, session, candidate)
