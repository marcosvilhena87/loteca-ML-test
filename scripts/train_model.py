#!/usr/bin/env python3
"""Train a lightweight probabilistic/structural model for Loteca picks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean

LOGGER = logging.getLogger(__name__)


def setup_logging(debug_path: Path) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(debug_path, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )


def laplace_prob(counts: dict, key: str, alpha: float = 1.0) -> float:
    total = sum(counts.values())
    classes = 3
    return (counts.get(key, 0) + alpha) / (total + alpha * classes)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_path = root / "output" / "processed_data.json"
    model_path = root / "models" / "model.json"
    debug_path = root / "output" / "debug.log"

    setup_logging(debug_path)
    if not processed_path.exists():
        raise FileNotFoundError("Execute scripts/preprocess_data.py antes do treinamento.")

    data = json.loads(processed_path.read_text(encoding="utf-8"))
    total_concursos = data["meta"]["total_concursos"]

    position_profiles = {}
    for pos in range(1, 15):
        key = str(pos)
        outcome_counts = data["position_outcome_counts"].get(key, {})
        top_counts = data["position_top_rank_hits"].get(key, {})

        position_profiles[key] = {
            "p_outcome_1": laplace_prob(outcome_counts, "1"),
            "p_outcome_X": laplace_prob(outcome_counts, "X"),
            "p_outcome_2": laplace_prob(outcome_counts, "2"),
            "p_top1_hit": laplace_prob(top_counts, "top1"),
            "p_top2_hit": laplace_prob(top_counts, "top2"),
            "p_top3_hit": laplace_prob(top_counts, "top3"),
        }

    top_count_series = data.get("per_concurso_top_counts", [])
    expected_top_hits = {
        "top1": mean([c.get("top1", 0) for c in top_count_series]) if top_count_series else 0.0,
        "top2": mean([c.get("top2", 0) for c in top_count_series]) if top_count_series else 0.0,
        "top3": mean([c.get("top3", 0) for c in top_count_series]) if top_count_series else 0.0,
    }

    architecture_counts = data.get("architecture_counts", {})
    top_vector_counts = data.get("top_vector_counts", {})
    model = {
        "meta": {
            "total_concursos": total_concursos,
            "top_architectures_sampled": len(architecture_counts),
            "top_vectors_sampled": len(top_vector_counts),
        },
        "position_profiles": position_profiles,
        "expected_top_hits": expected_top_hits,
        "most_frequent_architectures": sorted(
            architecture_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:10],
        "most_frequent_top_vectors": sorted(
            top_vector_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:10],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("Modelo treinado com %s concursos válidos.", total_concursos)
    LOGGER.info("Esperança estrutural de acertos: %s", expected_top_hits)
    LOGGER.info("Modelo salvo em %s", model_path)


if __name__ == "__main__":
    main()
