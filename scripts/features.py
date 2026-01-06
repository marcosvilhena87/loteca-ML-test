import re
from typing import Dict, List

import numpy as np
import pandas as pd

PROB_COLUMNS: List[str] = ['P(1)', 'P(X)', 'P(2)']
ODDS_COLUMNS: List[str] = ['Odds_1', 'Odds_X', 'Odds_2']

# Ordem utilizada para mapear os índices das probabilidades para resultados legíveis
INDEX_TO_RESULT = {0: '1', 1: 'X', 2: '2'}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove caracteres estranhos e espaços das colunas."""
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    return df


def ensure_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Garante as colunas de probabilidades a partir das odds."""
    if not all(col in df.columns for col in PROB_COLUMNS):
        if not all(col in df.columns for col in ODDS_COLUMNS):
            raise KeyError(f"Colunas de odds ausentes: {ODDS_COLUMNS}")

        for prob_col, odd_col in zip(PROB_COLUMNS, ODDS_COLUMNS):
            df[prob_col] = 1 / df[odd_col]

    prob_sum = df[PROB_COLUMNS].sum(axis=1)
    df[PROB_COLUMNS] = df[PROB_COLUMNS].div(prob_sum, axis=0)
    return df


def add_mando_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas do mando de campo."""
    if 'Mando' in df.columns:
        mando_clean = df['Mando'].fillna('').astype(str).str.lower().str.strip()
        df['is_neutro'] = (mando_clean == 'neutro').astype(int)
        df['home_adv'] = 1 - df['is_neutro']
    else:
        df['is_neutro'] = np.nan
        df['home_adv'] = np.nan
    return df


def add_goal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera métricas baseadas nos gols dos jogos já ocorridos."""
    if {'Gols_Home', 'Gols_Away'}.issubset(df.columns):
        df['goal_diff'] = df['Gols_Home'] - df['Gols_Away']
        df['total_goals'] = df['Gols_Home'] + df['Gols_Away']
        df['is_over_25'] = (df['total_goals'] >= 3).astype(int)
        df['is_btts'] = ((df['Gols_Home'] > 0) & (df['Gols_Away'] > 0)).astype(int)
    else:
        df['goal_diff'] = np.nan
        df['total_goals'] = np.nan
        df['is_over_25'] = np.nan
        df['is_btts'] = np.nan
    return df


def _split_last5(raw: str) -> List[str]:
    raw = raw.strip()
    if ',' in raw:
        return [item.strip() for item in raw.split(',') if item.strip()]
    if ' ' in raw:
        return [item.strip() for item in raw.split() if item.strip()]
    return [item.strip() for item in raw.split('-') if item.strip()]


def _parse_last5_entry(entry: str) -> Dict[str, float]:
    metrics = {
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'gf5': 0,
        'ga5': 0,
        'btts_matches': 0,
        'over25_matches': 0,
        'games': 0,
    }

    if not entry or not isinstance(entry, str):
        return metrics

    tokens = _split_last5(entry)
    metrics['games'] = len(tokens)

    for token in tokens:
        match = re.match(r'^(\d+)[-xX](\d+)$', token)
        if match:
            gf, ga = int(match.group(1)), int(match.group(2))
            metrics['gf5'] += gf
            metrics['ga5'] += ga
            if gf > ga:
                metrics['wins'] += 1
            elif gf == ga:
                metrics['draws'] += 1
            else:
                metrics['losses'] += 1
            if gf > 0 and ga > 0:
                metrics['btts_matches'] += 1
            if gf + ga >= 3:
                metrics['over25_matches'] += 1
            continue

        token_upper = token.upper()
        if token_upper in {'V', 'W'}:
            metrics['wins'] += 1
        elif token_upper == 'E':  # Empate / Draw
            metrics['draws'] += 1
        else:
            metrics['losses'] += 1

    return metrics


def add_last5_features(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    """Extrai features de forma recente das colunas de último cinco jogos."""
    def compute_features(series: pd.Series, prefix: str) -> pd.DataFrame:
        records = []
        for entry in series.fillna('').astype(str):
            metrics = _parse_last5_entry(entry)
            games = metrics['games'] or 5  # evita divisão por zero
            points = metrics['wins'] * 3 + metrics['draws']
            form = points / (games * 3)
            record = {
                f'{prefix}_points_last5': points,
                f'{prefix}_wins_last5': metrics['wins'],
                f'{prefix}_draws_last5': metrics['draws'],
                f'{prefix}_losses_last5': metrics['losses'],
                f'{prefix}_form_last5': form,
                f'{prefix}_gf5': metrics['gf5'],
                f'{prefix}_ga5': metrics['ga5'],
                f'{prefix}_goal_diff5': metrics['gf5'] - metrics['ga5'],
                f'{prefix}_btts_rate': metrics['btts_matches'] / games,
                f'{prefix}_over25_rate': metrics['over25_matches'] / games,
            }
            records.append(record)
        return pd.DataFrame(records)

    if home_col in df.columns:
        home_df = compute_features(df[home_col], 'home')
        df = pd.concat([df.reset_index(drop=True), home_df], axis=1)
    else:
        df['home_points_last5'] = np.nan
        df['home_wins_last5'] = np.nan
        df['home_draws_last5'] = np.nan
        df['home_losses_last5'] = np.nan
        df['home_form_last5'] = np.nan
        df['home_gf5'] = np.nan
        df['home_ga5'] = np.nan
        df['home_goal_diff5'] = np.nan
        df['home_btts_rate'] = np.nan
        df['home_over25_rate'] = np.nan

    if away_col in df.columns:
        away_df = compute_features(df[away_col], 'away')
        df = pd.concat([df.reset_index(drop=True), away_df], axis=1)
    else:
        df['away_points_last5'] = np.nan
        df['away_wins_last5'] = np.nan
        df['away_draws_last5'] = np.nan
        df['away_losses_last5'] = np.nan
        df['away_form_last5'] = np.nan
        df['away_gf5'] = np.nan
        df['away_ga5'] = np.nan
        df['away_goal_diff5'] = np.nan
        df['away_btts_rate'] = np.nan
        df['away_over25_rate'] = np.nan

    df['form_diff'] = df['home_form_last5'] - df['away_form_last5']
    df['gf5_diff'] = df['home_gf5'] - df['away_gf5']
    df['ga5_diff'] = df['home_ga5'] - df['away_ga5']
    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva métricas adicionais a partir das odds e probabilidades."""
    if all(col in df.columns for col in ODDS_COLUMNS):
        implied = sum(1 / df[col] for col in ODDS_COLUMNS)
        df['overround'] = implied - 1

        fav_idx = df[ODDS_COLUMNS].values.argmin(axis=1)
        dog_idx = df[ODDS_COLUMNS].values.argmax(axis=1)
        df['fav_index'] = fav_idx
        df['dog_index'] = dog_idx
    else:
        df['overround'] = np.nan
        df['fav_index'] = np.nan
        df['dog_index'] = np.nan

    pmax = df[PROB_COLUMNS].max(axis=1)
    p2nd = df[PROB_COLUMNS].apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    df['pmax'] = pmax
    df['p2nd'] = p2nd
    df['gap'] = df['pmax'] - df['p2nd']

    return df


FEATURE_COLUMNS: List[str] = [
    'P(1)', 'P(X)', 'P(2)',
    'overround', 'pmax', 'p2nd', 'gap', 'fav_index', 'dog_index',
    'is_neutro', 'home_adv',
    'goal_diff', 'total_goals', 'is_over_25', 'is_btts',
    'home_points_last5', 'home_form_last5', 'home_goal_diff5', 'home_btts_rate', 'home_over25_rate',
    'away_points_last5', 'away_form_last5', 'away_goal_diff5', 'away_btts_rate', 'away_over25_rate',
    'form_diff', 'gf5_diff', 'ga5_diff',
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica toda a engenharia de atributos esperada pelo modelo."""
    df = clean_column_names(df)
    df = ensure_probabilities(df)
    df = add_mando_features(df)
    df = add_goal_features(df)

    home_last5_col = None
    away_last5_col = None
    for candidate_home in ['Home_Last5_Home', 'last-5-h2h-mandante']:
        if candidate_home in df.columns:
            home_last5_col = candidate_home
            break
    for candidate_away in ['Away_Last5_Away', 'last-5-h2h-visitante']:
        if candidate_away in df.columns:
            away_last5_col = candidate_away
            break

    df = add_last5_features(df, home_last5_col or '', away_last5_col or '')
    df = add_odds_features(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df
