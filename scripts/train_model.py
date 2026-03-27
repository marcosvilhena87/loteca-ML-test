import argparse
import logging
from statistics import mean
from typing import Dict, List

from scripts.common import dump_json, normalize_row, read_csv_semicolon, setup_logging
from scripts.preprocess_data import compute_targets


def train(history_path: str, model_path: str) -> Dict[str, object]:
    rows = [normalize_row(row) for row in read_csv_semicolon(history_path)]
    targets = compute_targets(rows)

    model = {
        "meta": {
            "history_rows": len(rows),
            "unique_concursos": len({r.concurso for r in rows}),
        },
        "hit_rate": {
            "top1": mean([r.top1_hit for r in rows]) if rows else 0.0,
            "top2": mean([r.top2_hit for r in rows]) if rows else 0.0,
            "top3": mean([r.top3_hit for r in rows]) if rows else 0.0,
        },
        "targets": targets,
        "hard_constraints": {
            "top1": 9,
            "top2": 5,
            "top3": 4,
            "secos": 12,
            "triplos": 2,
        },
        "soft_slot_weights": {
            "top1": 1.0,
            "top2": 1.0,
            "top3": 1.15,
        },
    }
    dump_json(model_path, model)
    logging.info("Modelo treinado e salvo em %s", model_path)
    logging.info("Resumo do modelo: %s", model)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina modelo estratégico para Loteca")
    parser.add_argument("--history", default="data/concursos_anteriores.csv")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    train(args.history, args.model)
