import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd

from scripts.preprocess_data import add_target_column, preprocess_dataset
from scripts.predict_results import predict_probabilities, select_ticket
from scripts.train_model import metrics_to_dict, train_model


def setup_logging(run_dir: str) -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger("loteca")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline ML Loteca")
    parser.add_argument("--dataset", default="data/concursos_anteriores.csv")
    parser.add_argument("--upcoming", default="data/proximo_concurso.csv")
    parser.add_argument("--output", default="output/palpite.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="p13")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--validation_split", type=float, default=0.2)
    return parser.parse_args()


def log_dataset_stats(logger, stats, df):
    logger.info(
        "Dataset stats | total=%s valid=%s invalid_missing_odds=%s invalid_low_odds=%s invalid_target=%s",
        stats.total_rows,
        stats.valid_rows,
        stats.invalid_missing_odds,
        stats.invalid_low_odds,
        stats.invalid_target,
    )

    if "target" in df.columns:
        class_counts = df["target"].value_counts().to_dict()
        logger.info("Class distribution: %s", json.dumps(class_counts, ensure_ascii=False))

    if {"Concurso"}.issubset(df.columns):
        concurso_min = df["Concurso"].min()
        concurso_max = df["Concurso"].max()
        logger.info("Concurso range: %s-%s", concurso_min, concurso_max)

    odds_cols = ["Odds_1", "Odds_X", "Odds_2"]
    odds_stats = df[odds_cols].agg(["min", "median", "max"]).to_dict()
    logger.info("Odds stats: %s", json.dumps(odds_stats, ensure_ascii=False))

    if "overround" in df.columns:
        overround = df["overround"]
        logger.info(
            "Overround stats | min=%.3f median=%.3f max=%.3f",
            overround.min(),
            overround.median(),
            overround.max(),
        )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    logger = setup_logging(run_dir)

    logger.info("Run config: %s", json.dumps(vars(args), ensure_ascii=False))

    train_df, feature_cols, stats = preprocess_dataset(args.dataset, upcoming=False)
    train_df = add_target_column(train_df)
    log_dataset_stats(logger, stats, train_df)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "final_model.pkl")

    model, metrics, split_stats = train_model(
        train_df,
        feature_cols=feature_cols,
        validation_split=args.validation_split,
        seed=args.seed,
        class_weight="balanced",
        model_path=model_path,
    )

    logger.info("Split stats: %s", json.dumps(split_stats, ensure_ascii=False))
    logger.info("Validation metrics: %s", json.dumps(metrics_to_dict(metrics), ensure_ascii=False))
    logger.info("Features used: %s", ", ".join(feature_cols))
    logger.info(
        "Baseline market logloss: %.6f | Model logloss: %.6f | Delta: %.2f%%",
        metrics.market_logloss,
        metrics.logloss,
        metrics.logloss_delta_pct,
    )

    upcoming_df, _, upcoming_stats = preprocess_dataset(args.upcoming, upcoming=True)
    log_dataset_stats(logger, upcoming_stats, upcoming_df)

    upcoming_df = predict_probabilities(upcoming_df, model, feature_cols)
    upcoming_df = upcoming_df.sort_values(["Concurso", "Jogo"]).reset_index(drop=True)
    ticket_df, summary = select_ticket(
        upcoming_df,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    logger.info("Ticket summary: %s", json.dumps(summary, ensure_ascii=False))
    top_duplos = summary.get("top_duplos") or []
    if top_duplos:
        logger.info("Top-6 candidatos a duplo (rank por score_duplo):")
        for rank, row in enumerate(top_duplos, start=1):
            logger.info(
                "Rank %s | Jogo %s | top1=%s top2=%s p_top1=%.4f p_top2=%.4f "
                "margem=%.4f entropy_pred=%.4f overround=%.4f score_duplo=%.4f",
                rank,
                row["Jogo"],
                row["top1"],
                row["top2"],
                row["p_top1"],
                row["p_top2"],
                row["margem"],
                row["entropy_pred"],
                row["overround"],
                row["score_duplo"],
            )
    logger.info(
        "Duplos selected: %s",
        ", ".join(ticket_df.loc[ticket_df["tipo"] == "DUPLO", "Jogo"].astype(str).tolist()),
    )
    duplos_ranked = ticket_df.loc[ticket_df["tipo"] == "DUPLO"].sort_values(
        "score_duplo", ascending=False
    )
    if not duplos_ranked.empty:
        logger.info("Resumo duplos (rank por score_duplo):")
        for rank, (_, row) in enumerate(duplos_ranked.iterrows(), start=1):
            logger.info(
                "Rank %s | Jogo %s: %s vs %s | top1=%s top2=%s p_top1=%.4f p_top2=%.4f "
                "margem=%.4f entropy_pred=%.4f overround=%.4f score_duplo=%.4f",
                rank,
                row.get("Jogo"),
                row.get("Mandante"),
                row.get("Visitante"),
                row.get("top1"),
                row.get("top2"),
                row.get("p_top1"),
                row.get("p_top2"),
                row.get("margem"),
                row.get("entropy_pred"),
                row.get("overround"),
                row.get("score_duplo"),
            )

    secos = ticket_df.loc[ticket_df["tipo"] == "SECO"]
    duplos = ticket_df.loc[ticket_df["tipo"] == "DUPLO"]
    sum_p_top1_secos = float(secos["p_top1"].sum())
    sum_p_top2_duplos = float(duplos["p_top2"].sum())
    entropy_secos = float(secos["entropy_pred"].mean())
    entropy_duplos = float(duplos["entropy_pred"].mean())
    logger.info(
        "Sanidade P13 | soma_p_top1_secos=%.4f soma_p_top2_duplos=%.4f "
        "entropy_secos=%.4f entropy_duplos=%.4f",
        sum_p_top1_secos,
        sum_p_top2_duplos,
        entropy_secos,
        entropy_duplos,
    )

    audit_cols = [
        "Concurso",
        "Jogo",
        "Mandante",
        "Visitante",
        "p1",
        "px",
        "p2",
        "top1",
        "top2",
        "margem",
        "entropy_pred",
        "score_duplo",
        "tipo",
    ]

    audit_path = os.path.join(run_dir, "auditoria_proximo_concurso.csv")
    ticket_df.to_csv(audit_path, sep=";", index=False)

    output_cols = [
        "Jogo",
        "palpite",
        "tipo",
        "p1",
        "px",
        "p2",
        "top1",
        "top2",
        "margem",
        "entropy_pred",
        "score_duplo",
    ]

    output_df = ticket_df[output_cols].rename(
        columns={"entropy_pred": "entropia"}
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, sep=";", index=False)

    logger.info("Output saved to %s", args.output)


if __name__ == "__main__":
    main()
