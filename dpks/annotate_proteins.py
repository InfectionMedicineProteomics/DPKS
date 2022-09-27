from Bio import SeqIO
import gzip

def parse_fasta(fasta):
    fasta_dict = {}
    for record in SeqIO.parse(fasta, "fasta"):
        id = record.id
        accession = id.split('|')[1]
        name = id.split('|')[2]
        fasta_dict[accession] = name
    return fasta_dict 

def fasta_to_dict(fasta_path: str) -> dict:
    fasta_dict = {}
    if fasta_path.endswith('gz'):
        with gzip.open(fasta_path, "rt") as handle:
            fasta_dict = parse_fasta(handle)
    else:
        fasta_dict = parse_fasta(fasta_path)
    return fasta_dict



def get_protein_labels(accessions : list[str], fasta_path : str) -> list[str]:
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



    
    