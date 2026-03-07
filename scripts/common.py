import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean

LOGGER = logging.getLogger(__name__)

TIE_PRIORITY = {"1": 0, "2": 1, "X": 2}
RANK_TO_LABEL = {1: "top1", 2: "top2", 3: "top3"}


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_decimal(value: str) -> float:
    if value is None:
        return 0.0
    value = value.strip().replace(",", ".")
    if not value:
        return 0.0
    return float(value)


def load_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for raw in reader:
            row = dict(raw)
            for col in ["Concurso", "Jogo", "1", "X", "2", "top1", "top2", "top3"]:
                if col in row and row[col] != "":
                    row[col] = int(row[col])
            for col in ["p(1)", "p(x)", "p(2)", "p(top1)", "p(top2)", "p(top3)"]:
                if col in row:
                    row[col] = parse_decimal(row[col])
            rows.append(row)
    LOGGER.info("Loaded %s rows from %s", len(rows), path)
    return rows


def group_by_concurso(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["Concurso"]].append(row)
    for _, games in grouped.items():
        games.sort(key=lambda r: r["Jogo"])
    return grouped


def rank_symbols(row):
    probs = {"1": row["p(1)"], "X": row["p(x)"], "2": row["p(2)"]}
    ranked = sorted(probs.items(), key=lambda kv: (-kv[1], TIE_PRIORITY[kv[0]]))
    return [symbol for symbol, _ in ranked]


def run_stats(binary_list):
    if not binary_list:
        return {"avg_run_length": 0.0, "runs_count": 0.0}
    run_lengths = []
    current = 0
    for v in binary_list:
        if v == 1:
            current += 1
        elif current:
            run_lengths.append(current)
            current = 0
    if current:
        run_lengths.append(current)
    runs_count = len(run_lengths)
    avg_run_length = mean(run_lengths) if run_lengths else 0.0
    return {"avg_run_length": float(avg_run_length), "runs_count": float(runs_count)}


def dump_json(path: str, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
