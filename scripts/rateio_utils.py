"""Utilities for loading and normalizing rateio (payout) data."""

import math
import pandas as pd


def load_rateio(filepath: str) -> pd.DataFrame:
    """Load rateio CSV data and compute helper columns.

    The function normalizes column names to snake_case and derives:
    - ``Log_Rateio_14``: natural log of ``Rateio_14`` (0 mapped to 0).
    - ``Acumulou_14``: flag indicating accumulation when there were no winners.
    """

    df = pd.read_csv(filepath, delimiter=';', decimal='.')

    rename_map = {
        "Ganhadores 14 Acertos": "Ganhadores_14",
        "Rateio 14 Acertos": "Rateio_14",
        "Arrecadacao Total": "Arrecadacao_Total",
    }
    df = df.rename(columns=rename_map)

    # Padroniza colunas mesmo quando já vêm com sufixos em português
    if "Rateio_14" not in df.columns and "Rateio_14_Acertos" in df.columns:
        df["Rateio_14"] = df["Rateio_14_Acertos"]
    if "Rateio_13" not in df.columns and "Rateio_13_Acertos" in df.columns:
        df["Rateio_13"] = df["Rateio_13_Acertos"]
    if "Ganhadores_14" not in df.columns and "Ganhadores_14_Acertos" in df.columns:
        df["Ganhadores_14"] = df["Ganhadores_14_Acertos"]

    if "Rateio_14" not in df.columns:
        raise KeyError("Coluna 'Rateio_14' ausente nos dados de rateio.")

    df["Log_Rateio_14"] = df["Rateio_14"].apply(lambda x: math.log(x) if x > 0 else 0)
    df["Acumulou_14"] = (df["Ganhadores_14"] == 0).astype(int)

    return df
