from typing import Union

import numpy as np
import pandas as pd


def parse_diann(
    diann_file: Union[str, pd.DataFrame], diann_qvalue: float = 0.01
) -> pd.DataFrame:
    if isinstance(diann_file, str):

        quantification_file = pd.read_csv(diann_file, sep="\t")

    else:

        quantification_file = diann_file

    quantification_file = quantification_file[
        quantification_file["Q.Value"] <= diann_qvalue
    ]

    used_columns = []

    scoring_columns = []

    if ("Lib.PG.Q.Value" in quantification_file) or (
        "Lib.Q.Value" in quantification_file
    ):

        if "Lib.PG.Q.Value" not in quantification_file:

            quantification_file["Lib.PG.Q.Value"] = 0.0

        columns = [
            "Protein.Ids",
            "Genes",
            "Precursor.Id",
            "Modified.Sequence",
            "Precursor.Charge",
            "RT",
            "Precursor.Quantity",
            "Lib.PG.Q.Value",
            "Lib.Q.Value",
        ]

        scoring_columns = ["Lib.PG.Q.Value", "Lib.Q.Value"]

        for col in columns:

            if col in quantification_file:

                used_columns.append(col)

    else:

        columns = [
            "Protein.Ids",
            "Genes",
            "Precursor.Id",
            "Modified.Sequence",
            "Precursor.Charge",
            "RT",
            "Precursor.Quantity",
            "Global.PG.Q.Value",
            "Global.Q.Value",
        ]

        scoring_columns = ["Global.PG.Q.Value", "Global.Q.Value"]

        for col in columns:

            if col in quantification_file:

                used_columns.append(col)

    file_column = "Run"

    if "Run" in quantification_file:

        long_results = quantification_file[["Run"] + used_columns].copy()

    elif "File.Name" in quantification_file:

        file_column = "File.Name"

        long_results = quantification_file[["File.Name"] + used_columns].copy()

    # Removes file extension (mzml, d, etc) and then rebuilds sample name
    long_results[file_column] = long_results[file_column].str.rsplit(".", 1).str[0]

    retention_times = (
        long_results[
            [
                "Protein.Ids",
                "Genes",
                "Precursor.Id",
                "Modified.Sequence",
                "Precursor.Charge",
                "RT",
            ]
        ]
        .groupby(
            [
                "Protein.Ids",
                "Genes",
                "Precursor.Id",
                "Modified.Sequence",
                "Precursor.Charge",
            ]
        )
        .agg("median")
    )

    wide_results = (
        long_results.pivot(
            columns=file_column,
            index=[
                "Precursor.Id",
                "Protein.Ids",
                "Genes",
                "Modified.Sequence",
                "Precursor.Charge",
            ]
            + scoring_columns,
            values="Precursor.Quantity",
        )
        .reset_index()
        .copy()
    )

    wide_results.columns.name = None

    wide_results = (
        wide_results.set_index(
            [
                "Protein.Ids",
                "Genes",
                "Precursor.Id",
                "Modified.Sequence",
                "Precursor.Charge",
            ]
        )
        .join(retention_times)
        .reset_index()
    )

    if ("Lib.PG.Q.Value" in quantification_file) or (
        "Lib.Q.Value" in quantification_file
    ):

        wide_results.rename(
            columns={
                "Precursor.Id": "PrecursorId",
                "Protein.Ids": "Protein",
                "Genes": "ProteinLabel",
                "Modified.Sequence": "PeptideSequence",
                "Precursor.Charge": "Charge",
                "Lib.PG.Q.Value": "ProteinQValue",
                "Lib.Q.Value": "PeptideQValue",
                "RT": "RetentionTime",
            },
            inplace=True,
        )

    else:

        wide_results.rename(
            columns={
                "Precursor.Id": "PrecursorId",
                "Protein.Ids": "Protein",
                "Genes": "ProteinLabel",
                "Modified.Sequence": "PeptideSequence",
                "Precursor.Charge": "Charge",
                "Global.PG.Q.Value": "ProteinQValue",
                "Global.Q.Value": "PeptideQValue",
                "RT": "RetentionTime",
            },
            inplace=True,
        )

    wide_results.replace(0, np.nan, inplace=True)

    return wide_results
