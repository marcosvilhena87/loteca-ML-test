"""Feature engineering utilities for the Loteca pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ['1', 'X', '2']
ODDS_COLUMNS = ['Odds 1', 'Odds X', 'Odds 2']
PROB_COLUMNS = ['P(1)', 'P(X)', 'P(2)']
FORM_COLUMNS = ['Home_Last5_Home', 'Away_Last5_Away']
ALTERNATIVE_FORM_COLUMNS = {
    'last-5-h2h-mandante': 'Home_Last5_Home',
    'last-5-h2h-visitante': 'Away_Last5_Away',
}
HOME_VALUES = {'mandante', 'h', 'home'}
MARKET_MAP = {'P(1)': 1, 'P(X)': 0, 'P(2)': -1}

# Features consumed by the model
MODEL_FEATURES = [
    'Form_Diff_Last5',
    'Is_Home',
    'Home_Fav',
    'Market_vs_Form',
    'Prob_Gap',
    'Market_Pmax',
    'Market_Entropy',
    'Odds_Ratio_1_2',
    'Log_Odds_Ratio_1_2',
    'Home_Prob_Gap',
]

# Features used by the "market corrector" model (stacking on top of odds)
MARKET_CORRECTOR_FEATURES = [
    *PROB_COLUMNS,
    'Form_Diff_Last5',
    'Is_Home',
    'Home_Fav',
    'Market_vs_Form',
]


def compute_probabilities(df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Compute normalized implied probabilities from odds.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing odds or probability columns.
    force : bool, optional
        If True, probabilities are recalculated even if present.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``P(1)``, ``P(X)`` and ``P(2)`` normalized.
    """
    df = df.copy()
    has_probs = all(col in df.columns for col in PROB_COLUMNS)
    if not has_probs or force:
        missing_columns = [col for col in ODDS_COLUMNS if col not in df.columns]
        if missing_columns:
            raise KeyError(f"As seguintes colunas necessárias estão ausentes: {missing_columns}")

        df[ODDS_COLUMNS] = df[ODDS_COLUMNS].apply(pd.to_numeric, errors='coerce')
        df['P(1)'] = 1 / df['Odds 1']
        df['P(X)'] = 1 / df['Odds X']
        df['P(2)'] = 1 / df['Odds 2']
    else:
        df[PROB_COLUMNS] = df[PROB_COLUMNS].apply(pd.to_numeric, errors='coerce')

    prob_sum = df[PROB_COLUMNS].sum(axis=1)
    df[PROB_COLUMNS] = df[PROB_COLUMNS].div(prob_sum, axis=0)
    return df


def _standardize_form_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure form-related columns are present and numeric."""
    df = df.copy()
    for src, target in ALTERNATIVE_FORM_COLUMNS.items():
        if src in df.columns and target not in df.columns:
            df[target] = df[src]

    for col in FORM_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Loteca-specific features used across training and prediction."""
    df = _standardize_form_columns(df)
    df[ODDS_COLUMNS] = df[ODDS_COLUMNS].apply(pd.to_numeric, errors='coerce')

    df['Form_Diff_Last5'] = df['Home_Last5_Home'] - df['Away_Last5_Away']
    df['Is_Home'] = df['Mando'].astype(str).str.lower().isin(HOME_VALUES).astype(int)
    df['Home_Fav'] = ((df['Is_Home'] == 1) & (df['Odds 1'] < df['Odds 2'])).astype(int)

    market_fav = df[PROB_COLUMNS].idxmax(axis=1)
    df['Market_Signal'] = market_fav.map(MARKET_MAP).fillna(0)
    df['Market_vs_Form'] = df['Market_Signal'] * df['Form_Diff_Last5']
    df['Prob_Gap'] = df[PROB_COLUMNS].max(axis=1) - df[PROB_COLUMNS].median(axis=1)
    df['Market_Pmax'] = df[PROB_COLUMNS].max(axis=1)

    epsilon = 1e-12
    adjusted_probs = df[PROB_COLUMNS].clip(lower=epsilon)
    df['Market_Entropy'] = -np.sum(adjusted_probs * np.log(adjusted_probs), axis=1)

    df['Odds_Ratio_1_2'] = df['Odds 1'] / df['Odds 2']
    df['Log_Odds_Ratio_1_2'] = np.log(df['Odds_Ratio_1_2'].replace(0, np.nan)).fillna(0)
    df['Home_Prob_Gap'] = df['Is_Home'] * df['Prob_Gap']
    return df
