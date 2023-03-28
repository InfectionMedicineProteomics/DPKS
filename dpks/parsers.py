from typing import Union

import numpy as np
import pandas as pd


def parse_diann(diann_file: Union[str, pd.DataFrame], diann_qvalue: float = 0.01) -> pd.DataFrame:
    if isinstance(diann_file, str):

        quantification_file = pd.read_csv(diann_file, sep="\t")

    else:

        quantification_file = diann_file

    quantification_file = quantification_file[quantification_file["Q.Value"] <= diann_qvalue]

    if "Lib.PG.Q.Value" in quantification_file:

        columns = [
            "Protein.Ids",
            "Precursor.Id",
            "Modified.Sequence",
            "Precursor.Charge",
            "RT",
            "Precursor.Quantity",
            "Lib.PG.Q.Value",
            "Lib.Q.Value"
        ]

    else:

        columns = [
            "Protein.Ids",
            "Precursor.Id",
            "Modified.Sequence",
            "Precursor.Charge",
            "RT",
            "Precursor.Quantity",
            "Global.PG.Q.Value",
            "Global.Q.Value"
        ]

    long_results = quantification_file[
        ["Run"] + columns
        ].copy()

    # Removes file extension (mzml, d, etc) and then rebuilds sample name
    long_results["Run"] = long_results["Run"].str.rsplit(".", 1).str[0]

    retention_times = long_results[
        ['Protein.Ids', 'Precursor.Id', 'Modified.Sequence', 'Precursor.Charge', 'RT']
    ].groupby(
        [
            "Protein.Ids",
            "Precursor.Id",
            "Modified.Sequence",
            "Precursor.Charge"
        ]
    ).agg('median')

    wide_results = long_results.pivot(
        columns="Run",
        index=[
            "Precursor.Id",
            "Protein.Ids",
            "Modified.Sequence",
            "Precursor.Charge",
            "Lib.PG.Q.Value",
            "Lib.Q.Value"
        ],
        values="Precursor.Quantity").reset_index().copy()

    wide_results.columns.name = None

    wide_results = wide_results.set_index(
        ['Protein.Ids', 'Precursor.Id', 'Modified.Sequence', 'Precursor.Charge']
    ).join(
        retention_times
    ).reset_index()

    if "Lib.PG.Q.Value" in quantification_file:

        wide_results.rename(
            columns={
                "Precursor.Id": "PrecursorId",
                "Protein.Ids": "Protein",
                "Modified.Sequence": "PeptideSequence",
                "Precursor.Charge": "Charge",
                "Lib.PG.Q.Value": "ProteinQValue",
                "Lib.Q.Value": "PeptideQValue",
                "RT": "RetentionTime"
            },
            inplace=True
        )

    else:

        wide_results.rename(
            columns={
                "Precursor.Id": "PrecursorId",
                "Protein.Ids": "Protein",
                "Modified.Sequence": "PeptideSequence",
                "Precursor.Charge": "Charge",
                "Global.PG.Q.Value": "ProteinQValue",
                "Global.Q.Value": "PeptideQValue",
                "RT": "RetentionTime"
            },
            inplace=True
        )

    wide_results.replace(0, np.nan, inplace=True)

    return wide_results
