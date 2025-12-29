"""Utilitário para simulação Monte Carlo das apostas atuais.

Este script lê um arquivo de previsões com probabilidades normalizadas e a
coluna ``Aposta`` com a combinação de secos/duplos. A partir disso, realiza
simulações para estimar a distribuição de acertos e as probabilidades de 14 e
13 acertos. Caso a probabilidade de poucos acertos (0–5) seja alta ou a
variância do número de acertos esteja acima de um limite, o script sinaliza a
necessidade de reequilíbrio (menos duplos aleatórios, mais favoritos seguros).
"""

import argparse
import logging
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


OutcomeProbabilities = np.ndarray
OutcomeMask = np.ndarray


def _parse_aposta_values(aposta_values: Iterable[str]) -> OutcomeMask:
    """Converte a coluna ``Aposta`` em uma máscara booleana (jogo x resultado).

    Cada linha pode conter um resultado simples (``"1"``, ``"X"`` ou ``"2"``)
    ou um duplo (por exemplo, ``"1, X"``). Espaços são ignorados.
    """

    mapping = {"1": 0, "X": 1, "x": 1, "2": 2}
    aposta_list: List[List[bool]] = []

    for raw_value in aposta_values:
        if isinstance(raw_value, float) and np.isnan(raw_value):
            raise ValueError("Valor ausente na coluna 'Aposta'; não é possível simular.")

        picks = [part.strip() for part in str(raw_value).split(",") if part.strip()]
        mask = [False, False, False]

        for pick in picks:
            if pick not in mapping:
                raise ValueError(f"Valor de aposta inesperado: '{pick}'. Use 1, X ou 2.")
            mask[mapping[pick]] = True

        if not any(mask):
            raise ValueError(f"Linha de aposta vazia após parse: '{raw_value}'")

        aposta_list.append(mask)

    return np.array(aposta_list, dtype=bool)


def _load_probabilities(path: str) -> Dict[str, np.ndarray]:
    """Lê o CSV de previsões e retorna probabilidades normalizadas e máscaras."""

    df = pd.read_csv(path, delimiter=";", decimal=".")
    required_cols = {"P(1)", "P(X)", "P(2)", "Aposta"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no arquivo {path}: {sorted(missing)}")

    raw_probs = df[["P(1)", "P(X)", "P(2)"]].to_numpy(dtype=float)
    prob_sum = raw_probs.sum(axis=1, keepdims=True)
    if np.any(prob_sum <= 0):
        raise ValueError("Probabilidades não podem somar 0 ou valores negativos.")

    normalized = raw_probs / prob_sum
    apuesta_mask = _parse_aposta_values(df["Aposta"].to_list())

    return {"probabilities": normalized, "aposta_mask": apuesta_mask}


def _simulate_hits(
    probabilities: OutcomeProbabilities, aposta_mask: OutcomeMask, n_simulations: int, seed: int
) -> np.ndarray:
    """Executa simulações e retorna o vetor com acertos por simulação."""

    rng = np.random.default_rng(seed)
    n_matches = probabilities.shape[0]
    cumulative = np.cumsum(probabilities, axis=1)

    samples = np.empty((n_simulations, n_matches), dtype=int)
    for idx, cum_prob in enumerate(cumulative):
        samples[:, idx] = np.searchsorted(cum_prob, rng.random(n_simulations))

    hits = np.zeros(n_simulations, dtype=int)
    for match_idx in range(n_matches):
        hits += aposta_mask[match_idx, samples[:, match_idx]]

    return hits


def _log_percentiles(hits: np.ndarray) -> None:
    percentiles = [5, 25, 50, 75, 90, 95]
    values = np.percentile(hits, percentiles)
    percentiles_str = ", ".join(f"p{p}%={v:.0f}" for p, v in zip(percentiles, values))
    logging.info("Percentis de acertos simulados: %s", percentiles_str)


def run_monte_carlo(
    predictions_path: str,
    n_simulations: int = 20000,
    low_hits_threshold: float = 0.30,
    variance_threshold: float = 8.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Executa a simulação Monte Carlo e retorna métricas agregadas.

    Parameters
    ----------
    predictions_path : str
        Caminho para o CSV com colunas ``P(1)``, ``P(X)``, ``P(2)`` e ``Aposta``.
    n_simulations : int, default 20000
        Número de cenários simulados.
    low_hits_threshold : float, default 0.30
        Limite máximo aceitável para P(0–5 acertos). Acima dele o cartão é
        bloqueado para ajuste.
    variance_threshold : float, default 8.0
        Variância máxima aceitável do número de acertos. Acima dela a aposta é
        considerada desbalanceada.
    seed : int, default 42
        Semente de aleatoriedade.
    """

    data = _load_probabilities(predictions_path)
    probabilities = data["probabilities"]
    aposta_mask = data["aposta_mask"]

    hits = _simulate_hits(probabilities, aposta_mask, n_simulations, seed)
    n_matches = probabilities.shape[0]

    distribution = np.bincount(hits, minlength=n_matches + 1) / n_simulations
    mean_hits = float(np.mean(hits))
    variance = float(np.var(hits))
    prob_low_hits = float(distribution[:6].sum())
    prob_14 = float(distribution[n_matches])
    prob_13 = float(distribution[n_matches - 1]) if n_matches >= 1 else 0.0

    logging.info("Média de acertos: %.3f | Variância: %.3f", mean_hits, variance)
    logging.info(
        "Probabilidades simuladas: P(14)=%.4f | P(13)=%.4f | P(0–5)=%.4f",
        prob_14,
        prob_13,
        prob_low_hits,
    )
    _log_percentiles(hits)

    blocked_reasons: List[str] = []
    if prob_low_hits > low_hits_threshold:
        blocked_reasons.append(
            f"P(0–5 acertos)={prob_low_hits:.3f} acima do limite {low_hits_threshold:.3f}"
        )
    if variance > variance_threshold:
        blocked_reasons.append(
            f"Variância={variance:.3f} acima do limite {variance_threshold:.3f}"
        )

    if blocked_reasons:
        logging.error(
            "Cartão bloqueado: %s. Reequilibrar (menos duplos aleatórios, mais favoritos seguros).",
            "; ".join(blocked_reasons),
        )
    else:
        logging.info("Cartão aprovado dentro dos limites de risco.")

    return {
        "mean_hits": mean_hits,
        "variance": variance,
        "prob_14": prob_14,
        "prob_13": prob_13,
        "prob_0_5": prob_low_hits,
        "blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulação Monte Carlo das apostas atuais.")
    parser.add_argument("predictions_path", help="CSV de entrada com probabilidades normalizadas e coluna Aposta")
    parser.add_argument("--n-simulations", type=int, default=20000, help="Número de cenários simulados (default: 20000)")
    parser.add_argument(
        "--low-hits-threshold",
        type=float,
        default=0.30,
        help="Limite máximo para P(0–5 acertos) antes de bloquear o cartão (default: 0.30)",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=8.0,
        help="Limite máximo de variância aceitável no total de acertos (default: 8.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_monte_carlo(
        predictions_path=args.predictions_path,
        n_simulations=args.n_simulations,
        low_hits_threshold=args.low_hits_threshold,
        variance_threshold=args.variance_threshold,
        seed=args.seed,
    )

    if results["blocked"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
