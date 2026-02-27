#!/usr/bin/env python3
"""Generate Loteca palpite with 12 secos, 2 duplos, 0 triplos."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

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
        rows = list(reader)
    return sorted(rows, key=lambda r: int(r["Jogo"]))


def normalized_probs(row: dict) -> Dict[str, float]:
    implied = {
        "1": 1.0 / parse_decimal(row["Odds_1"]),
        "X": 1.0 / parse_decimal(row["Odds_X"]),
        "2": 1.0 / parse_decimal(row["Odds_2"]),
    }
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()}


def ordered_symbols(prob_map: Dict[str, float]) -> List[str]:
    return sorted(prob_map.keys(), key=lambda symbol: prob_map[symbol], reverse=True)


def double_token(a: str, b: str) -> str:
    pair = "".join(sorted((a, b), key=lambda s: "1X2".index(s)))
    if pair == "1X":
        return "1X"
    if pair == "12":
        return "12"
    if pair == "X2":
        return "X2"
    raise ValueError(f"Par inválido para duplo: {a}, {b}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    next_path = root / "data" / "proximo_concurso.csv"
    model_path = root / "models" / "model.json"
    out_path = root / "output" / "predictions.csv"
    debug_path = root / "output" / "debug.log"

    setup_logging(debug_path)
    if not model_path.exists():
        raise FileNotFoundError("Execute scripts/train_model.py antes de prever.")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    rows = read_rows(next_path)
    if len(rows) != 14:
        LOGGER.warning("Concurso de entrada possui %s jogos (esperado 14).", len(rows))

    per_match = []
    for row in rows:
        pos = str(int(row["Jogo"]))
        probs = normalized_probs(row)
        ordered = ordered_symbols(probs)
        top1, top2, top3 = ordered
        margin = probs[top1] - probs[top2]

        pos_profile = model["position_profiles"].get(pos, {})
        p_top1_hist = pos_profile.get("p_top1_hit", 0.5)
        p_top2_hist = pos_profile.get("p_top2_hit", 0.3)

        double_score = (1 - margin) * (1 + p_top2_hist) * (1 + (1 - p_top1_hist))
        per_match.append(
            {
                "concurso": row["Concurso"],
                "jogo": int(row["Jogo"]),
                "mandante": row["Mandante"],
                "visitante": row["Visitante"],
                "prob_1": probs["1"],
                "prob_X": probs["X"],
                "prob_2": probs["2"],
                "top1": top1,
                "top2": top2,
                "top3": top3,
                "double_score": double_score,
            }
        )

    double_candidates = sorted(per_match, key=lambda g: g["double_score"], reverse=True)[:2]
    double_games = {g["jogo"] for g in double_candidates}

    LOGGER.info("Duplos selecionados (12 secos + 2 duplos + 0 triplos): %s", sorted(double_games))

    r14_categorical = []
    r14_prob = []
    prediction_rows = []

    for game in sorted(per_match, key=lambda g: g["jogo"]):
        if game["jogo"] in double_games:
            guess = double_token(game["top1"], game["top2"])
            pick_type = "duplo"
        else:
            guess = game["top1"]
            pick_type = "seco"

        r14_categorical.append(guess)
        r14_prob.append(f"{game['top1']}>{game['top2']}>{game['top3']}")

        LOGGER.info(
            "Jogo %02d %s x %s | probs(1=%.3f X=%.3f 2=%.3f) top=%s/%s/%s | tipo=%s palpite=%s",
            game["jogo"],
            game["mandante"],
            game["visitante"],
            game["prob_1"],
            game["prob_X"],
            game["prob_2"],
            game["top1"],
            game["top2"],
            game["top3"],
            pick_type,
            guess,
        )

        prediction_rows.append(
            {
                "Concurso": game["concurso"],
                "Jogo": game["jogo"],
                "Mandante": game["mandante"],
                "Visitante": game["visitante"],
                "Top1": game["top1"],
                "Top2": game["top2"],
                "Top3": game["top3"],
                "Palpite": guess,
                "Tipo": pick_type,
            }
        )

    expected = model.get("expected_top_hits", {})
    total_picks = sum(2 if token in {"1X", "12", "X2"} else 3 if token == "1X2" else 1 for token in r14_categorical)
    secos = sum(1 for token in r14_categorical if token in {"1", "X", "2"})
    duplos = sum(1 for token in r14_categorical if token in {"1X", "12", "X2"})
    triplos = sum(1 for token in r14_categorical if token == "1X2")
    LOGGER.info("Arquitetura final: %s secos, %s duplos, %s triplos, %s picks totais", secos, duplos, triplos, total_picks)
    LOGGER.info("R14 categórico palpite: %s", r14_categorical)
    LOGGER.info("R14 probabilístico (top1/top2/top3): %s", r14_prob)
    LOGGER.info("Esperança histórica top hits por concurso: %s", expected)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["Concurso", "Jogo", "Mandante", "Visitante", "Top1", "Top2", "Top3", "Palpite", "Tipo"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(prediction_rows)

    LOGGER.info("Predições gravadas em %s", out_path)


if __name__ == "__main__":
    main()
