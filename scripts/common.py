import csv
from pathlib import Path
from typing import Dict, List, Tuple

TIE_PRIORITY = {"1": 3, "2": 2, "X": 1}
RANK_TO_INDEX = {"top1": 1, "top2": 2, "top3": 3}
INDEX_TO_RANK = {1: "top1", 2: "top2", 3: "top3"}


def parse_decimal(value: str) -> float:
    value = (value or "").strip()
    if not value:
        return 0.0
    return float(value.replace(",", "."))


def read_csv_semicolon(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def write_csv_semicolon(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def rank_symbols(p1: float, px: float, p2: float) -> List[Tuple[str, float]]:
    pairs = [("1", p1), ("X", px), ("2", p2)]
    return sorted(pairs, key=lambda x: (x[1], TIE_PRIORITY[x[0]]), reverse=True)


def enrich_probabilities(row: Dict[str, str]) -> Dict[str, object]:
    p1 = parse_decimal(row.get("p(1)", "0"))
    px = parse_decimal(row.get("p(x)", "0"))
    p2 = parse_decimal(row.get("p(2)", "0"))
    ranking = rank_symbols(p1, px, p2)
    top_symbols = {"top1": ranking[0][0], "top2": ranking[1][0], "top3": ranking[2][0]}
    top_probs = {"top1": ranking[0][1], "top2": ranking[1][1], "top3": ranking[2][1]}
    out = dict(row)
    out.update({
        "_p(1)": p1,
        "_p(x)": px,
        "_p(2)": p2,
        "_top_symbols": top_symbols,
        "_top_probs": top_probs,
    })
    return out


def detect_hit_rank(row: Dict[str, str], enriched: Dict[str, object]) -> int:
    result_symbol = ""
    if row.get("1", "0") == "1":
        result_symbol = "1"
    elif row.get("X", "0") == "1":
        result_symbol = "X"
    elif row.get("2", "0") == "1":
        result_symbol = "2"

    if not result_symbol:
        return 0

    top_symbols = enriched["_top_symbols"]
    for rank_idx in (1, 2, 3):
        if top_symbols[INDEX_TO_RANK[rank_idx]] == result_symbol:
            return rank_idx
    return 0


def run_features(binary: List[int]) -> Dict[str, float]:
    runs = []
    n = len(binary)
    i = 0
    while i < n:
        if binary[i] == 1:
            start = i
            while i < n and binary[i] == 1:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1

    if not runs:
        return {"avg_run_length": 0.0, "avg_run_count": 0.0, "avg_run_position": 0.0}

    avg_len = sum(length for _, length in runs) / len(runs)
    avg_count = float(len(runs))
    avg_pos = sum((start + 1) + ((length - 1) / 2) for start, length in runs) / len(runs)
    return {
        "avg_run_length": avg_len,
        "avg_run_count": avg_count,
        "avg_run_position": avg_pos,
    }
