import pandas as pd

def get_protein_labels(proteins):
    """Maps UniProt ID to more common names. P69905 --> HBA_HUMAN
    Args:
        proteins list: Input protein list.
    Returns:
        list: More common protein names.
    """
    human_proteome = pd.read_csv('DPKS/dpks/data/human_proteome.gz')
    
    def format_protein_name(protein):
        """
        formats protein name. 
        sp|XXXX|ALBU_HUMAN --> XXXX
        XXXX --> XXXX
        """
        if '|' in protein:
            protein = protein.split('|')[1]
        if '_' in protein:
            protein = protein.split('_')[0]

        return protein
    human_proteome['accession'] = human_proteome['accession'].map(format_protein_name)
    names = []
    for protein in proteins:
        m = human_proteome['trivname'].to_numpy()[human_proteome['accession'].to_numpy() == protein]
        if len(m) == 0:
            m = protein
        else: 
            m = m[0].split('_')[0]
        names.append(m)
    return names