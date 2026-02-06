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

    upcoming_df, _, upcoming_stats = preprocess_dataset(args.upcoming, upcoming=True)
    log_dataset_stats(logger, upcoming_stats, upcoming_df)

    upcoming_df = predict_probabilities(upcoming_df, model, feature_cols)
    ticket_df, summary = select_ticket(
        upcoming_df,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    logger.info("Ticket summary: %s", json.dumps(summary, ensure_ascii=False))
    logger.info(
        "Duplos selected: %s",
        ", ".join(ticket_df.loc[ticket_df["tipo"] == "DUPLO", "Jogo"].astype(str).tolist()),
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
