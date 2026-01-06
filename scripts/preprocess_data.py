import logging
import math
from copy import deepcopy
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


FEATURE_COLUMNS = [
    "P(1)",
    "P(X)",
    "P(2)",
    "market_entropy",
    "pmax",
    "spread_1_2",
    "draw_bias",
    "is_neutro",
    "home_last5_points",
    "home_last5_winrate",
    "home_last5_wins",
    "home_last5_draws",
    "home_last5_losses",
    "home_last5_missing",
    "away_last5_points",
    "away_last5_winrate",
    "away_last5_wins",
    "away_last5_draws",
    "away_last5_losses",
    "away_last5_missing",
    "home_points_last5",
    "away_points_last5",
    "home_goal_diff_last5",
    "away_goal_diff_last5",
    "home_winrate_last5",
    "away_winrate_last5",
    "history_points_diff",
    "history_goal_diff_diff",
    "home_history_missing",
    "away_history_missing",
]


def _normalize_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    prob_sum = df[["P(1)", "P(X)", "P(2)"]].sum(axis=1)
    df["P(1)"] = df["P(1)"] / prob_sum
    df["P(X)"] = df["P(X)"] / prob_sum
    df["P(2)"] = df["P(2)"] / prob_sum
    return df


def _derive_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df["market_entropy"] = -(
        df[["P(1)", "P(X)", "P(2)"]].mul(
            df[["P(1)", "P(X)", "P(2)"]].applymap(lambda p: math.log(max(p, 1e-15)))
        ).sum(axis=1)
    )
    df["pmax"] = df[["P(1)", "P(X)", "P(2)"]].max(axis=1)
    df["spread_1_2"] = (df["P(1)"] - df["P(2)"]).abs()
    df["draw_bias"] = df["P(X)"]
    return df


def _parse_last5(sequence: str) -> List[str]:
    if pd.isna(sequence):
        return []
    return [char for char in str(sequence) if char in {"V", "E", "D"}]


def _last5_metrics(sequence: str) -> Tuple[int, int, int, int, float, int]:
    results = _parse_last5(sequence)
    wins = results.count("V")
    draws = results.count("E")
    losses = results.count("D")
    points = wins * 3 + draws
    games = len(results)
    winrate = points / (games * 3) if games else 0.0
    missing_flag = int(games == 0)
    return points, wins, draws, losses, winrate, missing_flag


def _add_last5_features(
    df: pd.DataFrame, home_col: str, away_col: str
) -> pd.DataFrame:
    (
        df["home_last5_points"],
        df["home_last5_wins"],
        df["home_last5_draws"],
        df["home_last5_losses"],
        df["home_last5_winrate"],
        df["home_last5_missing"],
    ) = zip(*df[home_col].map(_last5_metrics))

    (
        df["away_last5_points"],
        df["away_last5_wins"],
        df["away_last5_draws"],
        df["away_last5_losses"],
        df["away_last5_winrate"],
        df["away_last5_missing"],
    ) = zip(*df[away_col].map(_last5_metrics))
    return df


def _calculate_points(goals_for: float, goals_against: float) -> int:
    if pd.isna(goals_for) or pd.isna(goals_against):
        return 0
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _update_history(
    history: Dict[str, List[Tuple[float, float, int]]],
    team: str,
    goals_for: float,
    goals_against: float,
    points: int,
):
    history.setdefault(team, []).append((goals_for, goals_against, points))


def _aggregate_history(
    history: Dict[str, List[Tuple[float, float, int]]], team: str, last_n: int
) -> Dict[str, float]:
    records = history.get(team, [])[-last_n:]
    games = len(records)
    goals_for = sum(g[0] for g in records)
    goals_against = sum(g[1] for g in records)
    points = sum(g[2] for g in records)
    return {
        "points": points,
        "goal_diff": goals_for - goals_against,
        "winrate": points / (games * 3) if games else 0.0,
        "games": games,
    }


def _add_team_history_features(
    df: pd.DataFrame,
    last_n: int = 5,
    base_history: Dict[str, List[Tuple[float, float, int]]] = None,
) -> pd.DataFrame:
    df = df.sort_values(["Concurso", "Jogo"]).reset_index(drop=True)
    history = deepcopy(base_history) if base_history is not None else {}

    home_points, away_points = [], []
    home_goal_diff, away_goal_diff = [], []
    home_winrate, away_winrate = [], []
    home_missing, away_missing = [], []

    for _, row in df.iterrows():
        mandante = row["Mandante"]
        visitante = row["Visitante"]

        home_stats = _aggregate_history(history, mandante, last_n)
        away_stats = _aggregate_history(history, visitante, last_n)

        home_points.append(home_stats["points"])
        away_points.append(away_stats["points"])
        home_goal_diff.append(home_stats["goal_diff"])
        away_goal_diff.append(away_stats["goal_diff"])
        home_winrate.append(home_stats["winrate"])
        away_winrate.append(away_stats["winrate"])
        home_missing.append(int(home_stats["games"] == 0))
        away_missing.append(int(away_stats["games"] == 0))

        # Evita vazamento: atualiza somente após computar features
        if not pd.isna(row.get("Gols_Home")) and not pd.isna(row.get("Gols_Away")):
            home_points_match = _calculate_points(row["Gols_Home"], row["Gols_Away"])
            away_points_match = _calculate_points(row["Gols_Away"], row["Gols_Home"])
            _update_history(history, mandante, row["Gols_Home"], row["Gols_Away"], home_points_match)
            _update_history(history, visitante, row["Gols_Away"], row["Gols_Home"], away_points_match)

    df["home_points_last5"] = home_points
    df["away_points_last5"] = away_points
    df["home_goal_diff_last5"] = home_goal_diff
    df["away_goal_diff_last5"] = away_goal_diff
    df["home_winrate_last5"] = home_winrate
    df["away_winrate_last5"] = away_winrate
    df["home_history_missing"] = home_missing
    df["away_history_missing"] = away_missing
    df["history_points_diff"] = df["home_points_last5"] - df["away_points_last5"]
    df["history_goal_diff_diff"] = df["home_goal_diff_last5"] - df["away_goal_diff_last5"]
    return df


def build_team_history(
    df: pd.DataFrame, last_n: int = 5
) -> Dict[str, List[Tuple[float, float, int]]]:
    """Generate historical stats per team to feed future feature creation."""

    df_sorted = df.sort_values(["Concurso", "Jogo"]).reset_index(drop=True)
    history: Dict[str, List[Tuple[float, float, int]]] = {}
    for _, row in df_sorted.iterrows():
        if pd.isna(row.get("Gols_Home")) or pd.isna(row.get("Gols_Away")):
            continue
        _update_history(
            history,
            row["Mandante"],
            row["Gols_Home"],
            row["Gols_Away"],
            _calculate_points(row["Gols_Home"], row["Gols_Away"]),
        )
        _update_history(
            history,
            row["Visitante"],
            row["Gols_Away"],
            row["Gols_Home"],
            _calculate_points(row["Gols_Away"], row["Gols_Home"]),
        )
    return history


def add_common_features(
    df: pd.DataFrame,
    last_n: int = 5,
    base_history: Dict[str, List[Tuple[float, float, int]]] = None,
) -> pd.DataFrame:
    """Create all model features without introducing data leakage."""

    odds_columns = ["Odds_1", "Odds_X", "Odds_2"]
    missing_columns = [col for col in odds_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"As seguintes colunas necessárias estão ausentes: {missing_columns}")

    home_last5_col = (
        "Home_Last5_Home"
        if "Home_Last5_Home" in df.columns
        else "last-5-h2h-mandante" if "last-5-h2h-mandante" in df.columns else None
    )
    away_last5_col = (
        "Away_Last5_Away"
        if "Away_Last5_Away" in df.columns
        else "last-5-h2h-visitante" if "last-5-h2h-visitante" in df.columns else None
    )
    if not home_last5_col or not away_last5_col:
        raise KeyError("Colunas de histórico recente (last5) não encontradas nos dados.")

    df = df.copy()
    df["P(1)"] = 1 / df["Odds_1"]
    df["P(X)"] = 1 / df["Odds_X"]
    df["P(2)"] = 1 / df["Odds_2"]

    df = _normalize_probabilities(df)
    df = _derive_market_features(df)

    df["is_neutro"] = df["Mando"].str.lower().ne("mandante").astype(int)

    df = _add_last5_features(df, home_last5_col, away_last5_col)
    df = _add_team_history_features(df, last_n=last_n, base_history=base_history)
    return df


def process(input_file: str, output_file: str, last_n: int = 5):
    """Clean historical data, engineer features and persist the training set."""

    try:
        logging.info("Carregando dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Gerando features...")
        df = add_common_features(df, last_n=last_n)

        logging.info("Determinando os resultados reais dos jogos...")
        if all(col in df.columns for col in ['[1]', '[x]', '[2]']):
            df['Resultado'] = df.apply(lambda row: '1' if row['[1]'] == 1 else
                                                   'X' if row['[x]'] == 1 else
                                                   '2' if row['[2]'] == 1 else None, axis=1)
        else:
            raise KeyError("As colunas '[1]', '[x]' e '[2]' são necessárias para calcular o resultado.")

        logging.info("Removendo linhas inválidas...")
        df = df.dropna(subset=['Resultado', 'P(1)', 'P(X)', 'P(2)'])

        logging.info(f"Salvando o arquivo processado em {output_file}...")
        df.to_csv(output_file, index=False, sep=';', decimal='.')
        logging.info("Processamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
