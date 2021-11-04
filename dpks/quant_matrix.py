from __future__ import annotations

import numpy as np
import networkx as nx

from csv import DictReader


class Protein:

    accession : str

    def __init__(self, accession : str = ''):

        self.accession = accession

class Precursor:

    peptide_sequence : str
    charge : int
    decoy : int
    retention_time : float
    index : int

    def __init__(self,
                 peptide_sequence : str = '',
                 charge : int = 0,
                 decoy : int = 0,
                 retention_time : float = 0.0,
                 index : int = 0):

        self.peptide_sequence = peptide_sequence
        self.charge = charge
        self.decoy = decoy
        self.retention_time = retention_time
        self.index = index

class Fragment:

    pass


class QuantMatrix:

    proteins : dict[str, Protein]
    precursors : dict[str, Precursor]
    fragments : dict[str, Fragment]
    matrix : np.ndarray
    quant_graph : nx.Graph
    design_matrix : list[dict[str, str]]

    def __init__(self, design_matrix : list = None, num_samples : int = 0, num_quant_records : int = 0):

        self.design_matrix = design_matrix

        self.proteins = dict()
        self.precursors = dict()
        self.fragments = dict()

        self.matrix = np.zeros(shape=(num_quant_records, num_samples), dtype='f8') # 64-bit floating-point number

        self.quant_graph = nx.Graph()


    @classmethod
    def from_csv(cls, file_path : str = '', design_matrix_path : str = '', quant_type : str = '') -> QuantMatrix:

        assert quant_type in ['fragment', 'precursor', 'peptide', 'protein']

        design_matrix = list()

        with open(design_matrix_path, 'r') as design_matrix_path_h:

            csv_reader = DictReader(design_matrix_path_h)

            for record in csv_reader:

                design_matrix_record = {
                    'name': record['Sample']
                }

                if 'Group' in record:

                    design_matrix_record['group'] = record['Group']

                if 'Batch' in record:

                    design_matrix_record['batch'] = record['Batch']

                design_matrix.append(design_matrix_record)

        with open(file_path, 'r') as quant_file_h:

            csv_reader = DictReader(quant_file_h)

            records = list(csv_reader)

            num_quant_records = len(records)

            quant_matrix = QuantMatrix(
                design_matrix=design_matrix,
                num_samples=len(design_matrix),
                num_quant_records=num_quant_records
            )

            for index, record in enumerate(records):

                if quant_type == 'precursor':

                    precursor_id = f"{record['PeptideSequence']}_{record['Charge']}"

                    precursor = Precursor(
                        peptide_sequence=record['PeptideSequence'],
                        charge=record['Charge'],
                        decoy=record['Decoy'],
                        retention_time=record['RetentionTime'],
                        index=index
                    )

                    quant_matrix.quant_graph.add_nodes_from(
                        [precursor_id],
                        data=precursor,
                        bipartite="precursor"
                    )

                    protein_accessions = record['Protein'].split(';')

                    for protein_accession in protein_accessions:

                        protein = Protein(
                            accession=protein_accession
                        )

                        if protein_accession not in quant_matrix.quant_graph:

                            quant_matrix.quant_graph.add_nodes_from(
                                [protein_accession],
                                data=protein,
                                bipartite="protein"
                            )

                        quant_matrix.quant_graph.add_edges_from(
                            [
                                (protein_accession, precursor_id)
                            ]
                        )

                    for sample_index, sample_info in enumerate(design_matrix):

                        sample_name = sample_info['name']

                        intensity = record[sample_name]

                        quant_matrix.matrix[index, sample_index] = intensity

        return quant_matrix




def create_quant_matrix(file_path : str = '', design_matrix : str = '', quant_type : str = '') -> QuantMatrix:

    print("Parsing Quant Matrix and building data structure.")

    return QuantMatrix.from_csv(file_path, design_matrix, quant_type)


if __name__ == '__main__':

    quant_file_path = '/home/aaron/projects/dpks/data/pyprophet_baseline_matrix.csv'
    design_matrix_file_path = '/home/aaron/projects/dpks/data/design_matrix.tsv'

    quant_matrix = create_quant_matrix(
        quant_file_path,
        design_matrix_file_path,
        quant_type='precursor'
    )