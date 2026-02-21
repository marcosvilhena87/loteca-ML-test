import argparse
import csv
import json
from pathlib import Path

OUTCOME_CLASSES = ("1", "X", "2")


def parse_decimal(value: str) -> float:
    value = (value or "").strip().replace(".", "").replace(",", ".")
    return float(value)


def infer_outcome(row: dict) -> str:
    for outcome in OUTCOME_CLASSES:
        if str(row.get(outcome, "0")).strip() == "1":
            return outcome
    return ""


def normalize_probs(o1: float, ox: float, o2: float) -> dict:
    inv = {
        "1": 1.0 / max(o1, 1e-9),
        "X": 1.0 / max(ox, 1e-9),
        "2": 1.0 / max(o2, 1e-9),
    }
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def preprocess(input_csv: Path, output_json: Path) -> None:
    processed = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            o1 = parse_decimal(row["Odds_1"])
            ox = parse_decimal(row["Odds_X"])
            o2 = parse_decimal(row["Odds_2"])
            probs = normalize_probs(o1, ox, o2)
            top_order = sorted(OUTCOME_CLASSES, key=lambda c: probs[c], reverse=True)
            processed.append(
                {
                    "concurso": int(row["Concurso"]),
                    "jogo": int(row["Jogo"]),
                    "mandante": row["Mandante"],
                    "visitante": row["Visitante"],
                    "odds": {"1": o1, "X": ox, "2": o2},
                    "probs": probs,
                    "top_order": top_order,
                    "outcome": infer_outcome(row),
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processamento Loteca")
    parser.add_argument("--input", default="data/concursos_anteriores.csv")
    parser.add_argument("--output", default="data/concursos_anteriores.processed.json")
    args = parser.parse_args()
    preprocess(Path(args.input), Path(args.output))
    print(f"Arquivo processado salvo em: {args.output}")
