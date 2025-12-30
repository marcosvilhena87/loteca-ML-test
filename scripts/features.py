import logging
import math
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market probabilities, margin and uncertainty signals from odds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain ``Odds 1``, ``Odds X`` and ``Odds 2`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns for market probabilities, margin, gap and entropy.
    """
    odds_columns = ["Odds 1", "Odds X", "Odds 2"]
    missing_columns = [col for col in odds_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"As seguintes colunas de odds são necessárias: {missing_columns}")

    inv_prob_cols = {"P_market(1)": 1 / df["Odds 1"], "P_market(X)": 1 / df["Odds X"], "P_market(2)": 1 / df["Odds 2"]}
    market_probs = pd.DataFrame(inv_prob_cols)
    margin = market_probs.sum(axis=1) - 1
    market_probs = market_probs.div(market_probs.sum(axis=1), axis=0)
    df = df.copy()
    for col in market_probs.columns:
        df[col] = market_probs[col]

    df["bookmaker_margin"] = margin
    df["top_prob_market"] = market_probs.max(axis=1)
    sorted_probs = market_probs.apply(lambda row: np.sort(row.values)[::-1], axis=1, result_type="expand")
    df["gap_market"] = sorted_probs[0] - sorted_probs[1]
    epsilon = 1e-12
    entropy = -np.sum(market_probs.clip(epsilon, 1).values * np.log(market_probs.clip(epsilon, 1).values), axis=1)
    df["entropia_market"] = entropy
    return df


class RatingEngine:
    """Incrementally compute Elo-like and Poisson-based features."""

    def __init__(self, home_advantage: float = 50.0, base_k: float = 24.0, decay_half_life: int = 150,
                 smoothing: float = 0.2, draw_scale: float = 300.0, base_draw: float = 0.22,
                 form_window: int = 5):
        self.home_advantage = home_advantage
        self.base_k = base_k
        self.decay_half_life = decay_half_life
        self.smoothing = smoothing
        self.draw_scale = draw_scale
        self.base_draw = base_draw
        self.form_window = form_window
        self.reset()

    def reset(self):
        self.elos: Dict[str, float] = defaultdict(lambda: 1500.0)
        self.game_counts: Dict[str, int] = defaultdict(int)
        self.elo_changes: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10))
        self.form_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.form_window))
        self.attack_strength: Dict[str, float] = defaultdict(lambda: 1.0)
        self.defense_weakness: Dict[str, float] = defaultdict(lambda: 1.0)
        self.global_attack = 1.0
        self.global_defense = 1.0
        self.total_matches = 0
        self.league_attack: Dict[str, float] = defaultdict(lambda: 1.0)
        self.league_defense: Dict[str, float] = defaultdict(lambda: 1.0)

    def _expected_home_win_prob(self, elo_diff: float) -> float:
        return 1 / (1 + 10 ** (-elo_diff / 400))

    def _dynamic_k(self, team: str) -> float:
        games = self.game_counts[team]
        decay = math.exp(-games / self.decay_half_life)
        return self.base_k * (0.5 + 0.5 * decay)

    def _update_elo(self, home: str, away: str, result: str, elo_diff: float):
        expected_home = self._expected_home_win_prob(elo_diff)
        actual_home = 1.0 if result == '1' else 0.5 if result == 'X' else 0.0
        change_home = self._dynamic_k(home) * (actual_home - expected_home)
        change_away = -self._dynamic_k(away) * (actual_home - expected_home)

        self.elos[home] += change_home
        self.elos[away] += change_away
        self.elo_changes[home].append(change_home)
        self.elo_changes[away].append(change_away)
        self.game_counts[home] += 1
        self.game_counts[away] += 1

        self.form_history[home].append(actual_home)
        self.form_history[away].append(1 - actual_home if result != 'X' else 0.5)

    def _update_poisson_rates(self, home: str, away: str, goals_home: float, goals_away: float,
                              league: Optional[str] = None):
        self.total_matches += 1
        global_smoothing = self.smoothing
        capped_home = min(goals_home, 5)
        capped_away = min(goals_away, 5)
        self.global_attack = (1 - global_smoothing) * self.global_attack + global_smoothing * max(capped_home, 0.1)
        self.global_defense = (1 - global_smoothing) * self.global_defense + global_smoothing * max(capped_away, 0.1)

        self.attack_strength[home] = (1 - self.smoothing) * self.attack_strength[home] + self.smoothing * max(capped_home, 0.1)
        self.attack_strength[away] = (1 - self.smoothing) * self.attack_strength[away] + self.smoothing * max(capped_away, 0.1)

        self.defense_weakness[home] = (1 - self.smoothing) * self.defense_weakness[home] + self.smoothing * max(capped_away, 0.1)
        self.defense_weakness[away] = (1 - self.smoothing) * self.defense_weakness[away] + self.smoothing * max(capped_home, 0.1)

        if league is not None:
            self.league_attack[league] = (1 - global_smoothing) * self.league_attack[league] + global_smoothing * max(capped_home, 0.1)
            self.league_defense[league] = (1 - global_smoothing) * self.league_defense[league] + global_smoothing * max(capped_away, 0.1)

    @staticmethod
    def _poisson_probs(lambda_home: float, lambda_away: float, max_goals: int = 6) -> Tuple[float, float, float]:
        def pmf(lmbd, k):
            return (lmbd ** k) * math.exp(-lmbd) / math.factorial(k)

        p_home = 0.0
        p_draw = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = pmf(lambda_home, i) * pmf(lambda_away, j)
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
        p_away = 1 - p_home - p_draw
        return p_home, p_draw, max(0.0, p_away)

    def _current_poisson_lambdas(self, home: str, away: str, league: Optional[str] = None) -> Tuple[float, float]:
        league_attack = self.league_attack.get(league, self.global_attack)
        league_defense = self.league_defense.get(league, self.global_defense)
        lambda_home = league_attack * (self.attack_strength[home] + self.defense_weakness[away]) / 2
        lambda_away = league_defense * (self.attack_strength[away] + self.defense_weakness[home]) / 2
        return max(lambda_home, 0.1), max(lambda_away, 0.1)

    def compute_match_features(self, row: pd.Series) -> Dict[str, float]:
        home, away = row["Mandante"], row["Visitante"]
        league = row.get("Liga") if "Liga" in row else row.get("Campeonato")
        elo_diff = (self.elos[home] + self.home_advantage) - self.elos[away]
        expected_home = self._expected_home_win_prob(elo_diff)
        draw_bias = self.base_draw * math.exp(-abs(elo_diff) / self.draw_scale)
        draw_prob = min(0.6, draw_bias)
        remaining = max(1e-6, 1 - draw_prob)
        p_home = expected_home * remaining
        p_away = (1 - expected_home) * remaining
        norm = p_home + p_away + draw_prob
        p_home /= norm
        p_away /= norm
        draw_prob /= norm

        elo_uncertainty_home = np.std(self.elo_changes[home]) if self.elo_changes[home] else 0.0
        elo_uncertainty_away = np.std(self.elo_changes[away]) if self.elo_changes[away] else 0.0

        form_home = np.mean(self.form_history[home]) if self.form_history[home] else 0.5
        form_away = np.mean(self.form_history[away]) if self.form_history[away] else 0.5

        lambda_home, lambda_away = self._current_poisson_lambdas(home, away, league)
        p_pois_home, p_pois_draw, p_pois_away = self._poisson_probs(lambda_home, lambda_away)

        return {
            "elo_diff": elo_diff,
            "elo_uncertainty_home": elo_uncertainty_home,
            "elo_uncertainty_away": elo_uncertainty_away,
            "form_home": form_home,
            "form_away": form_away,
            "form_diff": form_home - form_away,
            "P_elo(1)": p_home,
            "P_elo(X)": draw_prob,
            "P_elo(2)": p_away,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "P_pois(1)": p_pois_home,
            "P_pois(X)": p_pois_draw,
            "P_pois(2)": p_pois_away,
        }

    def update_after_match(self, row: pd.Series):
        home, away = row["Mandante"], row["Visitante"]
        league = row.get("Liga") if "Liga" in row else row.get("Campeonato")
        result = row.get("Resultado")
        goals_home = row.get("Gols_Home") if not math.isnan(row.get("Gols_Home", float("nan"))) else None
        goals_away = row.get("Gols_Away") if not math.isnan(row.get("Gols_Away", float("nan"))) else None

        elo_diff = (self.elos[home] + self.home_advantage) - self.elos[away]
        if result in {"1", "X", "2"}:
            self._update_elo(home, away, result, elo_diff)

        if goals_home is not None and goals_away is not None:
            self._update_poisson_rates(home, away, goals_home, goals_away, league)


def compute_expert_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta features highlighting deviations from the betting market."""

    required = [
        "P_market(1)", "P_market(X)", "P_market(2)",
        "P_elo(1)", "P_elo(X)", "P_elo(2)",
        "P_pois(1)", "P_pois(X)", "P_pois(2)",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Para calcular deltas vs. mercado, faltam as colunas: {missing}")

    df = df.copy()
    df["d_elo_1"] = df["P_elo(1)"] - df["P_market(1)"]
    df["d_elo_X"] = df["P_elo(X)"] - df["P_market(X)"]
    df["d_elo_2"] = df["P_elo(2)"] - df["P_market(2)"]

    df["d_pois_1"] = df["P_pois(1)"] - df["P_market(1)"]
    df["d_pois_X"] = df["P_pois(X)"] - df["P_market(X)"]
    df["d_pois_2"] = df["P_pois(2)"] - df["P_market(2)"]

    df["draw_boost"] = df["P_pois(X)"] - df["P_market(X)"]
    return df


def enrich_features(df: pd.DataFrame, engine: RatingEngine, update_results: bool = True) -> pd.DataFrame:
    """Add Elo/Poisson-derived features to the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least ``Mandante`` and ``Visitante`` columns.
    engine : RatingEngine
        Stateful engine reused across historical and future games.
    update_results : bool, optional
        If ``True``, updates internal ratings using the match outcomes.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with new feature columns.
    """
    enriched = []
    for _, row in df.iterrows():
        features = engine.compute_match_features(row)
        record = row.to_dict()
        record.update(features)
        enriched.append(record)
        if update_results:
            engine.update_after_match(row)
    return pd.DataFrame(enriched)
