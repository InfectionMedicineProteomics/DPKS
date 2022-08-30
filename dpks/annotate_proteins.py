import pandas as pd

def get_protein_labels(proteins):
    """Maps UniProt ID to more common names. P69905 --> HBA_HUMAN

    Args:
        proteins list: Input protein list.

    Returns:
        list: More common protein names.
    """
    human_proteome = pd.read_csv('data/human_proteome.gz')
    
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
    
    human_proteome['accession'] = human_proteome['accession'].apply(lambda x: format_protein_name(x))
    names = []
    for protein in proteins:
        if protein in human_proteome['accession'].values:
            m = human_proteome.loc[human_proteome['accession'] == protein]['trivname'].values
            assert len(m) == 1
            m = m[0].split('_')[0]
        else:
            m = protein
        names.append(m)
    return names

