"""Utilities for parsing Loteca rateio (prize) files."""
import numpy as np
import pandas as pd


def _winsorize(series: pd.Series, lower: float = 0.0, upper: float = 0.99) -> pd.Series:
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def load_rateio(path, include_extended: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', decimal='.', encoding='utf-8-sig')
    rename_map = {
        'Concurso': 'Concurso',
        'Ganhadores 14 Acertos': 'Ganhadores_14',
        'Rateio 14 Acertos': 'Rateio_14',
        'Ganhadores 13 Acertos': 'Ganhadores_13',
        'Rateio 13 Acertos': 'Rateio_13',
        'Arrecadacao Total': 'Arrecadacao_Total',
    }
    df = df.rename(columns=rename_map)
    df['Ganhadores_14'] = pd.to_numeric(df['Ganhadores_14'], errors='coerce').fillna(0).astype(int)
    df['Rateio_14'] = pd.to_numeric(df['Rateio_14'], errors='coerce')
    if 'Ganhadores_13' in df:
        df['Ganhadores_13'] = pd.to_numeric(df['Ganhadores_13'], errors='coerce').fillna(0).astype(int)
    else:
        df['Ganhadores_13'] = 0
    if 'Rateio_13' in df:
        df['Rateio_13'] = pd.to_numeric(df['Rateio_13'], errors='coerce')
    df['Arrecadacao_Total'] = pd.to_numeric(df['Arrecadacao_Total'], errors='coerce')

    df['Rateio_14_winsor'] = _winsorize(df['Rateio_14'].fillna(0))
    df['Rateio_13_winsor'] = _winsorize(df['Rateio_13'].fillna(0)) if 'Rateio_13' in df else pd.Series(dtype=float)

    df['Log_Rateio_14'] = df['Rateio_14'].replace(0, pd.NA).apply(lambda x: np.log(x) if pd.notna(x) else np.nan)
    df['Log_Rateio_14_winsor'] = df['Rateio_14_winsor'].replace(0, pd.NA).apply(lambda x: np.log(x) if pd.notna(x) else np.nan)
    df['Acumulou_14'] = (df['Ganhadores_14'] == 0).astype(int)

    base_columns = [
        'Concurso', 'Ganhadores_14', 'Rateio_14', 'Arrecadacao_Total', 'Log_Rateio_14', 'Acumulou_14'
    ]

    extended_columns = [
        'Rateio_14_winsor', 'Ganhadores_13', 'Rateio_13', 'Rateio_13_winsor', 'Log_Rateio_14_winsor'
    ]

    if include_extended:
        ordered_columns = base_columns + [c for c in extended_columns if c in df.columns]
    else:
        ordered_columns = base_columns

    return df[ordered_columns]
