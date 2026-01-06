"""Backtesting utilities for measuring accuracy and EV per concurso.

O script consome um CSV de predições (gerado pelo ``predict_results.py``),
compara com os resultados reais do concurso e estima um retorno simples usando
os rateios oficiais e o custo do cartão.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

from scripts.rateio_utils import load_rateio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _compute_result(goals_home: float, goals_away: float) -> str:
    if goals_home > goals_away:
        return "1"
    if goals_home < goals_away:
        return "2"
    return "X"


def _parse_aposta(aposta: str) -> List[str]:
    return [choice.strip() for choice in str(aposta).split(",")]


def _count_bet_types(pred_df: pd.DataFrame) -> Tuple[int, int, int]:
    secos = duplos = triplos = 0
    for aposta in pred_df["Aposta"]:
        choices = _parse_aposta(aposta)
        if len(choices) == 1:
            secos += 1
        elif len(choices) == 2:
            duplos += 1
        else:
            triplos += 1
    return secos, duplos, triplos


def _lookup_bet_cost(
    secos: int, duplos: int, triplos: int, valor_df: pd.DataFrame
) -> Tuple[int, float]:
    match = valor_df[
        (valor_df["Secos"] == secos)
        & (valor_df["Duplos"] == duplos)
        & (valor_df["Triplos"] == triplos)
    ]
    if match.empty:
        raise ValueError(
            f"Combinação de aposta não encontrada na tabela: {secos} secos, {duplos} duplos, {triplos} triplos"
        )

    row = match.iloc[0]
    return int(row["Nº de Apostas"]), float(row["Valor"])


def _evaluate_hits(merged_df: pd.DataFrame) -> int:
    def _hit(row) -> bool:
        choices = _parse_aposta(row["Aposta"])
        return row["Resultado"] in choices

    return int(merged_df.apply(_hit, axis=1).sum())


def backtest(
    predictions_file: str,
    resultados_file: str = "data/raw/concursos_anteriores.csv",
    rateio_file: str = "data/raw/concurso_rateio.csv",
    valor_cartao_file: str = "data/raw/valor_cartao.csv",
) -> Dict[str, float]:
    """Compute hit rate and EV for a single concurso prediction sheet."""

    logging.info("Carregando predições em %s", predictions_file)
    pred_df = pd.read_csv(predictions_file, delimiter=";", decimal=".")

    if "Concurso" not in pred_df.columns:
        raise KeyError("A coluna 'Concurso' é obrigatória para o backtest.")

    concurso_id = pred_df["Concurso"].iloc[0]
    logging.info("Concurso alvo: %s", concurso_id)

    logging.info("Carregando resultados reais...")
    resultados_df = pd.read_csv(resultados_file, delimiter=";", decimal=".")
    if "Resultado" not in resultados_df.columns:
        resultados_df["Resultado"] = resultados_df.apply(
            lambda r: _compute_result(r["Gols_Home"], r["Gols_Away"]), axis=1
        )

    resultados_df = resultados_df[resultados_df["Concurso"] == concurso_id]
    merged = pred_df.merge(
        resultados_df[["Concurso", "Jogo", "Resultado"]], on=["Concurso", "Jogo"], how="left"
    )

    if merged["Resultado"].isna().any():
        raise ValueError("Resultados reais ausentes para alguns jogos deste concurso.")

    total_hits = _evaluate_hits(merged)
    logging.info("Total de acertos: %d", total_hits)

    secos, duplos, triplos = _count_bet_types(pred_df)
    logging.info("Perfil da aposta — Secos: %d, Duplos: %d, Triplos: %d", secos, duplos, triplos)

    valor_df = pd.read_csv(valor_cartao_file, delimiter=";", decimal=".")
    n_apostas, cost = _lookup_bet_cost(secos, duplos, triplos, valor_df)
    logging.info("Número de apostas geradas: %d | Custo total: %.2f", n_apostas, cost)

    rateio_df = load_rateio(rateio_file)
    rateio_row = rateio_df[rateio_df["Concurso"] == concurso_id]
    if rateio_row.empty:
        raise ValueError(f"Rateio não encontrado para o concurso {concurso_id}.")

    rateio_row = rateio_row.iloc[0]
    payout = 0.0
    if total_hits >= 14:
        payout = float(rateio_row["Rateio_14"])
    elif total_hits == 13 and "Rateio_13" in rateio_row.index:
        payout = float(rateio_row["Rateio_13"])

    ev = payout - cost

    summary = {
        "concurso": concurso_id,
        "acertos": total_hits,
        "n_apostas": n_apostas,
        "custo": cost,
        "retorno": payout,
        "ev": ev,
    }

    logging.info("Resumo do backtest: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Roda um backtest simples por concurso")
    parser.add_argument("predictions_file", help="CSV gerado pelo predict_results.py")
    parser.add_argument(
        "--resultados_file",
        default="data/raw/concursos_anteriores.csv",
        help="CSV com resultados históricos",
    )
    parser.add_argument(
        "--rateio_file",
        default="data/raw/concurso_rateio.csv",
        help="CSV com rateio oficial do concurso",
    )
    parser.add_argument(
        "--valor_cartao_file",
        default="data/raw/valor_cartao.csv",
        help="Tabela de custos por combinação de secos/duplos/triplos",
    )

    args = parser.parse_args()
    backtest(
        args.predictions_file,
        resultados_file=args.resultados_file,
        rateio_file=args.rateio_file,
        valor_cartao_file=args.valor_cartao_file,
    )
