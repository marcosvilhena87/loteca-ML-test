"""Utilities for parsing Loteca rateio (prize) files and sampling distributions."""
import numpy as np
import pandas as pd


def _winsorize(series: pd.Series, lower: float = 0.0, upper: float = 0.99) -> pd.Series:
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def _sample_series(series: pd.Series, n_samples: int) -> np.ndarray:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.zeros(n_samples)
    return np.random.choice(cleaned, size=n_samples, replace=True)


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
    df['Ganhadores_13'] = pd.to_numeric(df.get('Ganhadores_13', pd.Series(dtype=float)), errors='coerce').fillna(0).astype(int)
    df['Rateio_13'] = pd.to_numeric(df.get('Rateio_13', pd.Series(dtype=float)), errors='coerce')
    df['Arrecadacao_Total'] = pd.to_numeric(df['Arrecadacao_Total'], errors='coerce')

    df['Rateio_14_winsor'] = _winsorize(df['Rateio_14'].fillna(0))
    df['Rateio_13_winsor'] = _winsorize(df['Rateio_13'].fillna(0))

    df['Log_Rateio_14'] = df['Rateio_14'].replace(0, pd.NA).apply(lambda x: np.log(x) if pd.notna(x) else np.nan)
    df['Log_Rateio_14_winsor'] = df['Rateio_14_winsor'].replace(0, pd.NA).apply(lambda x: np.log(x) if pd.notna(x) else np.nan)
    df['Acumulou_14'] = (df['Ganhadores_14'] == 0).astype(int)

    base_columns = [
        'Concurso', 'Ganhadores_14', 'Rateio_14', 'Arrecadacao_Total', 'Log_Rateio_14', 'Acumulou_14'
    ]

    extended_columns = [
        'Ganhadores_13', 'Rateio_13', 'Rateio_14_winsor', 'Rateio_13_winsor', 'Log_Rateio_14_winsor'
    ]

    if include_extended:
        ordered_columns = base_columns + [c for c in extended_columns if c in df.columns]
    else:
        ordered_columns = base_columns

    return df[ordered_columns]


def sample_rateio_distribution(rateio_df: pd.DataFrame, n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return bootstrap samples for rateios 14/13 using winsorized history.

    The Monte Carlo sampling guards against over-optimism from a single extreme
    rateio by resampling plausible historical payouts. If winsorized columns are
    missing, the raw values are used as fallback.
    """

    rateio_14 = rateio_df.get('Rateio_14_winsor', rateio_df.get('Rateio_14', pd.Series(dtype=float)))
    rateio_13 = rateio_df.get('Rateio_13_winsor', rateio_df.get('Rateio_13', pd.Series(dtype=float)))

    return _sample_series(rateio_14, n_samples), _sample_series(rateio_13, n_samples)
