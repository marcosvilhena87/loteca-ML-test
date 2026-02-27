#!/usr/bin/env python3
"""Preprocess historical Loteca data into feature/statistics artifacts."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

LOGGER = logging.getLogger(__name__)
OUTCOMES = ("1", "X", "2")


def setup_logging(debug_path: Path) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(debug_path, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )


def parse_decimal(value: str) -> float:
    return float(value.strip().replace(",", "."))


def read_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def true_outcome(row: dict) -> str:
    for key in OUTCOMES:
        if row.get(key, "0").strip() == "1":
            return key
    raise ValueError(f"Linha sem resultado válido: concurso={row.get('Concurso')} jogo={row.get('Jogo')}")


def normalized_probs(row: dict) -> Dict[str, float]:
    implied = {
        "1": 1.0 / parse_decimal(row["Odds_1"]),
        "X": 1.0 / parse_decimal(row["Odds_X"]),
        "2": 1.0 / parse_decimal(row["Odds_2"]),
    }
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()}


def ordered_outcomes_by_prob(prob_map: Dict[str, float]) -> List[str]:
    return sorted(prob_map.keys(), key=lambda symbol: prob_map[symbol], reverse=True)


def build_processed(rows: List[dict]) -> dict:
    by_concurso: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_concurso[row["Concurso"].strip()].append(row)

    pos_outcome_counts = {str(i): Counter() for i in range(1, 15)}
    pos_top_rank_hit = {str(i): Counter() for i in range(1, 15)}
    architecture_counter = Counter()
    top_vector_counter = Counter()
    per_concurso_top_counts: List[dict] = []

    for concurso, games in by_concurso.items():
        sorted_games = sorted(games, key=lambda g: int(g["Jogo"]))
        if len(sorted_games) != 14:
            LOGGER.warning("Concurso %s ignorado por conter %s jogos (esperado: 14).", concurso, len(sorted_games))
            continue

        result_vector: List[str] = []
        top_vector: List[str] = []
        top_hit_counter = Counter()

        for game in sorted_games:
            pos = str(int(game["Jogo"]))
            outcome = true_outcome(game)
            probs = normalized_probs(game)
            ranking = ordered_outcomes_by_prob(probs)
            rank = ranking.index(outcome) + 1

            result_vector.append(outcome)
            top_vector.append(f"top{rank}")

            pos_outcome_counts[pos][outcome] += 1
            pos_top_rank_hit[pos][f"top{rank}"] += 1
            top_hit_counter[f"top{rank}"] += 1

        architecture_counter["".join(result_vector)] += 1
        top_vector_counter["".join(top_vector)] += 1
        per_concurso_top_counts.append(dict(top_hit_counter))

    LOGGER.info("Concursos válidos processados: %s", sum(architecture_counter.values()))

    return {
        "meta": {"total_concursos": sum(architecture_counter.values())},
        "position_outcome_counts": {k: dict(v) for k, v in pos_outcome_counts.items()},
        "position_top_rank_hits": {k: dict(v) for k, v in pos_top_rank_hit.items()},
        "architecture_counts": dict(architecture_counter),
        "top_vector_counts": dict(top_vector_counter),
        "per_concurso_top_counts": per_concurso_top_counts,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "concursos_anteriores.csv"
    output_path = root / "output" / "processed_data.json"
    debug_path = root / "output" / "debug.log"

    setup_logging(debug_path)
    LOGGER.info("Iniciando preprocessamento de %s", data_path)

    rows = read_rows(data_path)
    processed = build_processed(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Artifact salvo em %s", output_path)


if __name__ == "__main__":
    main()
