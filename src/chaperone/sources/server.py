"""Local MCP server exposing Human Protein Atlas lookups as tools.

Run standalone for a smoke test: `python -m chaperone.sources.server`
Consumed by chaperone.agent over stdio via the MCP client.
"""
from mcp.server.fastmcp import FastMCP

from . import cellphonedb_client, chembl_client, string_client
from .hpa_client import fetch_gene_profile
from .pubmed_client import pubmed_search
from .uniprot_client import fetch_uniprot_annotation

mcp = FastMCP("hpa")


@mcp.tool()
def hpa_expression(gene: str) -> dict:
    """Tissue and single-cell-type expression for a human gene, per the
    Human Protein Atlas. Only tissues/cell types where the gene is
    "enriched" (specifically elevated) are listed — this is a specificity
    profile, not a full quantitative expression matrix across all
    tissues/cell types.

    Args:
        gene: HGNC gene symbol, e.g. "EGFR".
    """
    profile = fetch_gene_profile(gene)
    if "error" in profile:
        return profile
    return {
        "gene": profile.get("Gene"),
        "rna_tissue_specific_nTPM": profile.get("RNA tissue specific nTPM"),
        "rna_tissue_specificity": profile.get("RNA tissue specificity"),
        "rna_single_cell_type_specific_nCPM": profile.get(
            "RNA single cell type specific nCPM"
        ),
        "rna_single_cell_type_specificity": profile.get(
            "RNA single cell type specificity"
        ),
        "protein_tissue_specific_intensity": profile.get(
            "Protein tissue specific Intensity"
        ),
    }


@mcp.tool()
def hpa_subcellular_location(gene: str) -> dict:
    """Annotated subcellular compartment(s) for a human gene's protein
    product, per the Human Protein Atlas immunofluorescence data.

    Args:
        gene: HGNC gene symbol, e.g. "EGFR".
    """
    profile = fetch_gene_profile(gene)
    if "error" in profile:
        return profile
    return {
        "gene": profile.get("Gene"),
        "subcellular_main_location": profile.get("Subcellular main location"),
        "subcellular_additional_location": profile.get(
            "Subcellular additional location"
        ),
        "subcellular_location": profile.get("Subcellular location"),
        "reliability_if": profile.get("Reliability (IF)"),
    }


@mcp.tool()
def hpa_protein_class(gene: str) -> dict:
    """Functional/structural protein class annotations for a human gene
    (e.g. "secreted", "membrane", "transcription factor"), per the Human
    Protein Atlas. Useful for deciding whether two proteins are expected
    to interact intracellularly (need coexpression in the same cell) or
    as a secreted-ligand/membrane-receptor pair (coexpression across
    different cells is fine).

    Args:
        gene: HGNC gene symbol, e.g. "EGFR".
    """
    profile = fetch_gene_profile(gene)
    if "error" in profile:
        return profile
    return {
        "gene": profile.get("Gene"),
        "gene_description": profile.get("Gene description"),
        "protein_class": profile.get("Protein class"),
        "secretome_location": profile.get("Secretome location"),
        "secretome_function": profile.get("Secretome function"),
        "known_interactions_count": profile.get("Interactions"),
    }


@mcp.tool()
def pubmed_ptm_glycosylation(gene: str) -> dict:
    """Search PubMed for reports that this gene's protein is glycosylated or
    carries other post-translational modifications (PTMs). AF3 predicts from
    bare, unmodified sequence — it does not model glycans and (by default)
    does not model most PTMs. Heavy glycosylation is common on secreted/
    membrane extracellular domains and can sterically block a modeled contact
    surface in reality; a PTM-dependent interaction (e.g. phosphorylation-
    gated binding) may only occur in a specific modification state AF3 didn't
    represent. Use this as a caveat signal, not a hard filter — it reports
    whether such literature exists for the gene generally, not whether the
    specific AF3-modeled interface residues are affected.

    Args:
        gene: HGNC gene symbol, e.g. "EGFR".
    """
    term = (
        f'"{gene}"[tiab] AND (glycosylation[tiab] OR glycosylated[tiab] OR '
        f'"N-linked"[tiab] OR "O-linked"[tiab] OR phosphorylation[tiab] OR '
        f'phosphorylated[tiab] OR "post-translational modification"[tiab])'
    )
    return pubmed_search(term)


@mcp.tool()
def pubmed_cooccurrence(gene_a: str, gene_b: str) -> dict:
    """Search PubMed for papers that mention both gene symbols together, as a
    weak proxy for whether this pair's interaction (or at least co-mention)
    is already reported in the literature. A high count doesn't confirm a
    known direct PPI (could be co-mentioned for unrelated reasons, e.g. same
    pathway/disease); a count of 0 is stronger evidence of genuine novelty.

    Args:
        gene_a: HGNC gene symbol of the first protein.
        gene_b: HGNC gene symbol of the second protein.
    """
    term = f'"{gene_a}"[tiab] AND "{gene_b}"[tiab]'
    return pubmed_search(term)


@mcp.tool()
def string_known_interaction(gene_a: str, gene_b: str) -> dict:
    """Check STRING (string-db.org) for a known/predicted functional
    association between two human gene symbols. Returns a combined score and
    a per-evidence-channel breakdown. A high "database" or "experiments"
    score is strong direct evidence; a high "textmining" score alone means
    the pair is discussed together in the literature, which is the same weak
    co-mention caveat as pubmed_cooccurrence, not confirmed direct binding.

    Args:
        gene_a: HGNC gene symbol of the first protein.
        gene_b: HGNC gene symbol of the second protein.
    """
    return string_client.known_interaction(gene_a, gene_b)


@mcp.tool()
def cellphonedb_known_interaction(gene_a: str, gene_b: str) -> dict:
    """Check CellPhoneDB's curated cell-cell-communication interactions
    (ligand-receptor pairs, adhesion/receptor complexes) for this gene pair.
    This is manually curated and specific to cell-surface/secreted signalling
    — a hit here is strong, precise evidence of an already-known interaction,
    especially relevant for a secreted-ligand/membrane-receptor candidate. A
    miss does NOT mean the pair is implausible — CellPhoneDB only covers
    cell-communication molecules, not general intracellular PPIs.

    Args:
        gene_a: HGNC gene symbol of the first protein.
        gene_b: HGNC gene symbol of the second protein.
    """
    return cellphonedb_client.known_interaction(gene_a, gene_b)


@mcp.tool()
def cellphonedb_complex_members(gene: str) -> dict:
    """Look up other gene symbols annotated alongside this one in a
    CellPhoneDB curated multi-subunit complex (e.g. integrin heterodimers,
    receptor complexes). Use this for the "other subunits" check — if this
    candidate's partner already forms a known complex missing from the AF3
    model, that's a LIKELY_SUBCOMPLEX signal. Only covers cell-surface/
    secreted complexes CellPhoneDB curates, not general intracellular ones.

    Args:
        gene: HGNC gene symbol.
    """
    return cellphonedb_client.complex_members(gene)


@mcp.tool()
def uniprot_annotation(gene: str) -> dict:
    """Look up a human gene's real UniProt entry: protein name, its FUNCTION
    description in plain text, keywords, and counts of PTM/topology feature
    types (Glycosylation, Lipidation, Disulfide bond, Transmembrane, etc.).

    Use this to check the protein's actual documented binding MECHANISM, not
    just whether a specific residue happens to carry a PTM. Example: a
    galectin's function text literally says "Lectin that binds beta-
    galactoside..." — that's a structural, mechanism-level reason AF3's
    bare-sequence fold may not represent the real interface at all
    (glycan-mediated, not protein-protein), independent of which residues
    the model happens to place in contact. Other useful keyword categories:
    "PTM" (e.g. Glycoprotein, Phosphoprotein), "Domain" (e.g. protein-protein
    interaction domains), "Ligand" (e.g. Calcium-binding, Zinc).

    Args:
        gene: HGNC gene symbol, e.g. "LGALS1".
    """
    return fetch_uniprot_annotation(gene)


# ---------------------------------------------------------------------------
# The small-molecule side of a protein-ligand candidate. STRING, CellPhoneDB
# and the HPA tools are all protein-to-protein, so without these a metabolite
# candidate would be triaged on the fold's confidence alone — which is the one
# thing that does not discriminate for a small molecule in a pocket.
# ---------------------------------------------------------------------------


@mcp.tool()
def resolve_small_molecule(name: str) -> dict:
    """Resolve a metabolite or drug name to a structure: ChEMBL id, SMILES
    and InChIKey, from ChEMBL and falling back to PubChem.

    Call this first for any small-molecule partner — the ChEMBL id it returns
    is what chembl_known_activity and chembl_ligand_promiscuity need, and an
    unresolvable name means the candidate cannot be folded as drawn.

    Args:
        name: a metabolite or drug name, e.g. "serotonin", "kynurenic acid".
    """
    return chembl_client.resolve_molecule(name)


@mcp.tool()
def chembl_known_activity(uniprot_accession: str, molecule_chembl_id: str) -> dict:
    """Curated ChEMBL bioactivities (Ki, Kd, IC50, EC50) of one small
    molecule at one protein target — the pharmacology literature's own record
    of whether this pair is already known.

    This is the small-molecule analogue of string_known_interaction and
    cellphonedb_known_interaction, neither of which indexes a metabolite
    against a receptor. A measured affinity is strong ALREADY_KNOWN evidence.
    Activities recorded against a protein FAMILY target are excluded, since
    they do not say which member was measured.

    Args:
        uniprot_accession: the receptor's accession, e.g. "P08908".
        molecule_chembl_id: from resolve_small_molecule, e.g. "CHEMBL39".
    """
    return chembl_client.known_activity(uniprot_accession, molecule_chembl_id)


@mcp.tool()
def chembl_ligand_promiscuity(molecule_chembl_id: str) -> dict:
    """How many DISTINCT protein targets this small molecule already has
    measured affinity at.

    Read this before treating a confident fold as evidence of a specific
    interaction. Co-folding confidence does not discriminate for a small
    molecule in a pocket: in this project's own orphan screen, 12 of 21
    metabolite candidates reached interface pTM 0.5 against EVERY receptor
    they were folded against, where no protein partner of 419 did. A molecule
    already measured at dozens of unrelated targets is a promiscuous binder,
    and a confident fold against one more receptor adds nothing.

    Args:
        molecule_chembl_id: from resolve_small_molecule, e.g. "CHEMBL39".
    """
    return chembl_client.promiscuity(molecule_chembl_id)


if __name__ == "__main__":
    mcp.run()
