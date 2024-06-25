import time
from typing import List

import pandas as pd
from Bio import SeqIO
import gzip

from unipressed import IdMappingClient

def parse_fasta(fasta):
    fasta_dict = {}
    for record in SeqIO.parse(fasta, "fasta"):
        id = record.id

        if "|" in id:
            accession = id.split("|")[1]
            name = id.split("|")[2]
        else:
            accession = id
            name = id
        fasta_dict[accession] = name
    return fasta_dict


def fasta_to_dict(fasta_path: str) -> dict:
    fasta_dict = {}
    if fasta_path.endswith("gz"):
        with gzip.open(fasta_path, "rt") as handle:
            fasta_dict = parse_fasta(handle)
    else:
        fasta_dict = parse_fasta(fasta_path)
    return fasta_dict


def get_protein_labels(accessions: List[str], fasta_path: str) -> List[str]:
    """Maps UniProt ID (accession) to more common names. P69905 --> HBA_HUMAN
    Args:
        accessions list[str]: Input protein list
        fasta_path str: fasta file path used for mapping
    Returns:
        protein_labels list[str]: More common protein names. If accession not in human_proteome.gz, use accession.
    """
    fasta_dict = fasta_to_dict(fasta_path)
    protein_labels = []
    for accession in accessions:
        try:
            protein_label = fasta_dict[accession]
        except KeyError:
            # If accession not in fasta file, put accession instead of protein label
            protein_label = accession
        protein_labels.append(protein_label)
    return protein_labels


def get_genes_from_proteins(proteins: List[str]) -> pd.DataFrame:

    request = IdMappingClient.submit(
        source="UniProtKB_AC-ID", dest="Gene_Name", ids=proteins
    )

    while True:
        status = request.get_status()
        if status in {"FINISHED", "ERROR"}:
            break
        else:
            time.sleep(1)

    translation_result = list(request.each_result())

    id_mapping = dict()

    for id_result in translation_result:
        mapping = id_mapping.get(id_result["from"], [])

        mapping.append(id_result["to"])

        id_mapping[id_result["from"]] = mapping

    final_mapping = dict()

    for key, value in id_mapping.items():
        value = value[0]

        final_mapping[key] = value

    mapping_df = pd.DataFrame(
        {"Protein": final_mapping.keys(), "Gene": final_mapping.values()}
    )

    return mapping_df