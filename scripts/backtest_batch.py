"""Backtests múltiplos concursos e compara estratégias de duplos."""

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from scripts.backtest import backtest
from scripts.predict_results import predict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _prepare_future_contest(historical_df: pd.DataFrame, concurso: int) -> pd.DataFrame:
    concurso_df = historical_df[historical_df["Concurso"] == concurso]
    if concurso_df.empty:
        raise ValueError(f"Concurso {concurso} não encontrado no histórico.")

    cleaned = concurso_df.drop(
        columns=["Gols_Home", "Gols_Away", "[1]", "[x]", "[2]", "Resultado"],
        errors="ignore",
    ).copy()
    return cleaned.sort_values("Jogo")


def _aggregate_results(rows: List[Dict]) -> pd.DataFrame:
    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    summary_df = (
        results_df.groupby("strategy")
        .agg(
            concursos_testados=("concurso", "count"),
            ev_medio=("ev", "mean"),
            perc_ev_positivo=("ev", lambda s: (s > 0).mean()),
            media_w14=("w14", "mean"),
            media_w13=("w13", "mean"),
        )
        .reset_index()
    )
    return summary_df


def backtest_batch(
    model_file: str,
    historical_file: str = "data/raw/concursos_anteriores.csv",
    rateio_file: str = "data/raw/concurso_rateio.csv",
    valor_cartao_file: str = "data/raw/valor_cartao.csv",
    output_dir: str = "output/backtest_batch",
    duplo_strategies: Iterable[str] = ("entropy", "top_margin"),
):
    logging.info("Carregando concursos anteriores...")
    historical_df = pd.read_csv(historical_file, delimiter=";", decimal=".")

    concursos = sorted(pd.unique(historical_df["Concurso"]))
    logging.info("Total de concursos disponíveis: %d", len(concursos))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    for concurso in concursos:
        logging.info("Processando concurso %s...", concurso)
        history_until_previous = historical_df[historical_df["Concurso"] < concurso]
        if history_until_previous.empty:
            logging.info(
                "Sem histórico anterior ao concurso %s — pulando backtest combinatório.",
                concurso,
            )
            continue

        future_df = _prepare_future_contest(historical_df, concurso)
        future_file = output_path / f"proximo_concurso_{concurso}.csv"
        future_df.to_csv(future_file, sep=";", index=False, decimal=".")

        for strategy in duplo_strategies:
            predictions_file = output_path / f"predictions_{strategy}_{concurso}.csv"
            logging.info(
                "Gerando predições com estratégia '%s' para o concurso %s...",
                strategy,
                concurso,
            )
            predict(
                str(future_file),
                model_file,
                str(predictions_file),
                duplo_strategy=strategy,
                historical_file=historical_file,
                historical_df=history_until_previous,
                history_max_concurso=concurso,
            )

            logging.info("Rodando backtest combinatório...")
            summary = backtest(
                str(predictions_file),
                resultados_file=historical_file,
                rateio_file=rateio_file,
                valor_cartao_file=valor_cartao_file,
            )
            summary["strategy"] = strategy
            results.append(summary)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning("Nenhum resultado gerado — verifique os dados de entrada.")
        return

    detailed_file = output_path / "backtest_batch_results.csv"
    results_df.to_csv(detailed_file, sep=";", index=False, decimal=".")
    logging.info("Resultados detalhados salvos em %s", detailed_file)

    summary_df = _aggregate_results(results)
    summary_file = output_path / "backtest_batch_summary.csv"
    summary_df.to_csv(summary_file, sep=";", index=False, decimal=".")
    logging.info("Resumo agregado salvo em %s", summary_file)

    logging.info("Resumo agregado:\n%s", summary_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Roda backtest em lote comparando estratégias de duplos"
    )
    parser.add_argument("model_file", help="Caminho para o modelo treinado")
    parser.add_argument(
        "--historical_file",
        default="data/raw/concursos_anteriores.csv",
        help="CSV com concursos anteriores (odds + resultados)",
    )
    parser.add_argument(
        "--rateio_file",
        default="data/raw/concurso_rateio.csv",
        help="CSV com rateios oficiais por concurso",
    )
    parser.add_argument(
        "--valor_cartao_file",
        default="data/raw/valor_cartao.csv",
        help="Tabela de custos por perfil de aposta",
    )
    parser.add_argument(
        "--output_dir",
        default="output/backtest_batch",
        help="Pasta para salvar predições e resumos",
    )
    parser.add_argument(
        "--duplo_strategies",
        nargs="+",
        default=["entropy", "top_margin"],
        help="Estratégias de duplo a serem comparadas",
    )

    args = parser.parse_args()

    backtest_batch(
        args.model_file,
        historical_file=args.historical_file,
        rateio_file=args.rateio_file,
        valor_cartao_file=args.valor_cartao_file,
        output_dir=args.output_dir,
        duplo_strategies=args.duplo_strategies,
    )
