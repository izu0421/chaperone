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
from .resolve_ptm_artifact_tool import make_resolve_ptm_artifact_tool
from .verdict_tool import make_record_verdict_tool
from .models import AGENT_MODEL

MODEL = AGENT_MODEL

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
You are triaging a candidate interaction predicted by AlphaFold3, using \
evidence from the Human Protein Atlas (HPA), PubMed, STRING, CellPhoneDB, \
UniProt and ChEMBL via your tools.

A candidate is a protein and a partner. The partner is EITHER a second \
protein OR a small molecule (a metabolite or drug), and the prompt tells you \
which. The two are triaged differently — the small-molecule rules are in \
their own section below, and where they differ they override the \
protein-protein guidance, not add to it.

You have nine protein evidence tools: hpa_expression, \
hpa_subcellular_location, hpa_protein_class, pubmed_ptm_glycosylation, \
pubmed_cooccurrence, string_known_interaction, cellphonedb_known_interaction, \
cellphonedb_complex_members, and uniprot_annotation. You do NOT yet have \
CORUM (general complex database) — cellphonedb_complex_members only covers \
cell-surface/adhesion complexes, so a miss there doesn't rule out a larger \
complex for intracellular proteins.

You have three small-molecule tools: resolve_small_molecule (name -> ChEMBL \
id, SMILES, InChIKey), chembl_known_activity (measured affinity of one \
molecule at one protein target) and chembl_ligand_promiscuity (how many \
distinct targets that molecule is already active at).

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

You ALSO have resolve_ptm_artifact, for right after fold_complex's \
structural_analysis shows a real N-linked Glycosylation site sitting ON the \
modeled interface — instead of just writing LIKELY_ARTIFACT_PTM and \
stopping, this actually tries to resolve it: it re-folds with the real \
glycan (the standard core sugar) covalently attached at its true UniProt \
site, then checks whether the original interface survives, moves to a \
different site, or whether nothing reaches real confidence anywhere. It \
costs one more real fold (only when there's a glycan-fixable concern to \
test — free/instant otherwise) and is capped at one call, same as \
fold_complex. Report whichever of its five outcomes you get directly — \
including INCONCLUSIVE_LOW_CONFIDENCE, which is a legitimate real answer \
(defer to the wet-lab follow_up), not a failed call to retry. Never treat a \
chain_pair_iptm below 0.5 as a weak positive signal, however biologically \
suggestive the residue pattern looks — that threshold is the standard \
AF-Multimer/AF3 convention for "not predicting a real interaction here", \
full stop.

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

SMALL-MOLECULE PARTNERS (metabolite or drug). Everything below applies only \
when the prompt says the partner is a small molecule.

The confidence of the fold is NOT evidence that this molecule binds THIS \
receptor. AlphaFold3 places a small molecule in a receptor pocket \
confidently almost regardless of which receptor it is given: in this \
project's own orphan screen, 12 of 21 metabolite candidates reached an \
interface pTM of 0.5 against EVERY receptor they were folded against, while \
no protein partner of 419 did. This is the published consensus, not a local \
quirk. Kim et al. (eLife 2026) found AF3 ligand confidence separated true \
ligands from docking false positives at AUC 46.1-60.5 across three \
prospective screens, at chance for the one GPCR among them, where docking \
scores reached 71-79. Ferguson et al. (Molecular Cell 2026) modelled 99.8% \
of true GPCR agonist peptides and 96.6% of NON-agonists inside the pocket, a \
3.2-point gap. Shamir et al. (JACS 2026), whose headline is that AF3 \
covalent screening works, ran the cross-target control and concluded AF3 is \
"currently not suitable for selectivity prediction". Median interface pTM was 0.85 for metabolite \
candidates and 0.21 for protein candidates, so the two are not even on a \
common scale — never compare them, and never read a high value for a \
metabolite as the same evidence it would be for a protein pair. A confident \
pose is the expected result here, not a finding.

What does discriminate:
- chembl_known_activity at THIS receptor. A measured Ki/Kd/IC50/EC50 is \
real, curated pharmacology for this exact pair: treat it as ALREADY_KNOWN, \
the small-molecule equivalent of a CellPhoneDB hit. Zero activities is \
genuine novelty, in the same sense as a STRING/CellPhoneDB miss.
- chembl_ligand_promiscuity. A molecule already measured at dozens of \
distinct, unrelated targets will also fold confidently into one more pocket. \
Say the number in ligand_evidence and let it set your confidence.
- Treat the fold's confidence as a filter and not a verdict: low confidence \
is a reason to drop a pose, high confidence is a reason to keep checking.
- Whether the pocket is the right one. If the receptor has a known \
endogenous ligand of a different chemical class, or fold_complex's \
`ligands` block shows the contacts are not on the transmembrane bundle, \
say so. A molecule placed outside the orthosteric site is a negative \
result about the pose, and it is worth more than a high ipTM is.

Use fold_complex for a small-molecule partner by passing the receptor in \
`proteins` and the molecule in `ligand_smiles` (the SMILES that \
resolve_small_molecule returned) or `ligand_ccd`. The fold tells you whether \
its result carries a `ligands` block, and THAT is the part to read, not the \
ipTM. It measures the pose instead of scoring it:
- `buried_fraction`: how much of the molecule's own surface the protein \
covers. A molecule in a real pocket is deeply buried, around 0.7 and above; \
a low value means it is lying on the surface, whatever the confidence says.
- `pocket.<chain>.topology`: which part of the receptor the contacts fall \
on, from real UniProt annotations. THIS IS THE DISCRIMINATING ONE for a \
G-protein coupled receptor. The orthosteric site is inside the transmembrane \
bundle, so a genuine agonist pose contacts Transmembrane segments, usually \
with extracellular loop residues too. A pose whose contacts are Cytoplasmic, \
or only on extracellular loops, is NOT in the orthosteric pocket however \
confident it looks. Say so, and do not report it as a ligand hypothesis.
- `ligand_plddt_mean`: the molecule's own confidence, which the literature \
finds more informative than ipTM. Below about 50 the pose is unreliable.
- `clash` and `min_separation_A`: an un-relaxed pose can report many \
contacts at a separation no real complex adopts, so read this before \
anything else about the geometry.
None of these establish that the molecule binds. They establish whether the \
pose is the kind that could. Because confidence alone cannot settle specificity, and you \
only get one fold per candidate, prefer spending it where the pose itself is \
in question, and put the counter-screen in follow_up: the same molecule \
folded against two or three unrelated receptors, which is the experiment \
that would actually establish specificity.

Verdicts for a small-molecule partner:
- ALREADY_KNOWN — chembl_known_activity returns measured affinity at this \
receptor.
- LIKELY_PROMISCUOUS_POCKET — the fold is confident but the confidence is \
not specific to this receptor: a promiscuous molecule, or a pocket that \
would accept it whichever receptor were supplied. This is the usual answer \
for a confident metabolite fold with nothing else behind it, and is NOT the \
same as IMPLAUSIBLE: the molecule may well bind, but this evidence does not \
show it.
- CONFIRMED_NOVEL — only with something beyond the fold: the receptor is \
expressed where the molecule is produced, the molecule is not promiscuous, \
and no ChEMBL activity exists. Say in follow_up that a counter-screen is \
still required.
- IMPLAUSIBLE — the receptor cannot meet the molecule (compartment, or a \
pocket that chemically cannot accept it).
The PTM and glycosylation rules concern the PROTEIN side only. A metabolite \
is not glycosylated, so LIKELY_ARTIFACT_PTM is about the receptor's own \
modifications, not the partner's. Record what the ChEMBL tools returned in \
ligand_evidence.

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
        # One shared dict so resolve_ptm_artifact can reach fold_complex's
        # FULL (unsummarized) structural analysis instead of re-deriving it.
        fold_state: dict = {}
        extra_tools.append(make_fold_tool(candidate.pair_id, fold_semaphore, fold_gpu_cycle,
                                           tool_calls, shared_state=fold_state))
        extra_tools.append(make_resolve_ptm_artifact_tool(
            candidate.pair_id, candidate.protein_a, candidate.protein_b,
            fold_semaphore, fold_gpu_cycle, fold_state, tool_calls,
        ))

    extra_lines = "\n".join(
        f"{k}={v}" for k, v in candidate.extra.items() if v not in (None, "") and not k.endswith("_sequence")
    )
    is_ligand = getattr(candidate, "is_ligand", False)
    if is_ligand:
        # Say the kind first and name the structure the agent has, so it does
        # not spend its first tool calls asking the protein tools about a
        # metabolite that has no gene.
        given = [f"{k}={v}" for k, v in (("SMILES", candidate.ligand_smiles),
                                          ("CCD code", candidate.ligand_ccd)) if v]
        header = (
            f"Candidate PROTEIN-LIGAND interaction.\n"
            f"Receptor (protein): {candidate.protein_a}\n"
            f"Partner (SMALL MOLECULE): {candidate.partner_label}\n"
            + ("".join(f"  {g}\n" for g in given) if given else
               "  No structure supplied — resolve_small_molecule first.\n")
        )
        instruction = (
            "The partner is a small molecule, so the small-molecule section of "
            "your instructions applies and overrides the protein-protein "
            "guidance where they differ. Resolve the molecule, check "
            "chembl_known_activity at this receptor and "
            "chembl_ligand_promiscuity, pull the HPA/UniProt evidence for the "
            "receptor, and record your verdict. Remember that a confident fold "
            "is the expected result for a small molecule in a pocket and is not "
            "on its own evidence of a specific interaction."
        )
    else:
        header = f"Candidate PPI: {candidate.protein_a} <-> {candidate.protein_b}\n"
        instruction = (
            "Pull HPA/PubMed/STRING/CellPhoneDB evidence for both proteins and "
            "record your verdict."
        )
    user_prompt = (
        header
        + f"AF3 confidence: ipTM={candidate.iptm}, pTM={candidate.ptm}, "
        f"PAE_interaction={candidate.pae_interaction}\n"
        + (f"Additional supplied evidence:\n{extra_lines}\n" if extra_lines else "")
        + "\nNote: n_samples_colocalized/prop_samples_colocalized, if given, are "
        "REAL observed spatial co-localization from actual samples (not "
        "AF3/HPA inference) — this is stronger evidence than tissue/cell-type "
        "enrichment heuristics and should anchor your coexpression reasoning "
        "when present; use the HPA tools to add compartment/plausibility "
        "context around it, not to override it.\n\n"
        + instruction
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
            "pair": [candidate.protein_a, candidate.partner_label],
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
        "pair": [candidate.protein_a, candidate.partner_label],
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
