import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

OUTCOME_PRIORITY = {"1": 2, "2": 1, "X": 0}


@dataclass(frozen=True)
class MatchRow:
    concurso: str
    jogo: int
    mandante: str
    visitante: str
    data: str
    p1: float
    px: float
    p2: float
    p_top1: float
    p_top2: float
    p_top3: float
    top1_hit: int
    top2_hit: int
    top3_hit: int


@dataclass(frozen=True)
class RunStats:
    avg_length: float
    avg_count: float
    avg_position: float


TOP_SLOT_NAMES = ["top1", "top2", "top3"]


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_decimal(value: str) -> float:
    text = (value or "").strip().replace(".", "").replace(",", ".")
    if not text:
        return 0.0
    return float(text)


def read_csv_semicolon(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def write_csv_semicolon(path: str, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_outcomes(p1: float, px: float, p2: float) -> List[Tuple[str, float]]:
    options = [("1", p1), ("X", px), ("2", p2)]
    return sorted(options, key=lambda item: (item[1], OUTCOME_PRIORITY[item[0]]), reverse=True)


def normalize_row(raw: Dict[str, str]) -> MatchRow:
    p1 = parse_decimal(raw.get("p(1)", "0"))
    px = parse_decimal(raw.get("p(x)", "0"))
    p2 = parse_decimal(raw.get("p(2)", "0"))
    ranked = rank_outcomes(p1, px, p2)
    return MatchRow(
        concurso=(raw.get("Concurso") or "").strip(),
        jogo=int((raw.get("Jogo") or "0").strip() or 0),
        mandante=(raw.get("Mandante") or "").strip(),
        visitante=(raw.get("Visitante") or "").strip(),
        data=(raw.get("Data") or "").strip(),
        p1=p1,
        px=px,
        p2=p2,
        p_top1=ranked[0][1],
        p_top2=ranked[1][1],
        p_top3=ranked[2][1],
        top1_hit=int((raw.get("top1") or "0").strip() or 0),
        top2_hit=int((raw.get("top2") or "0").strip() or 0),
        top3_hit=int((raw.get("top3") or "0").strip() or 0),
    )


def runs_metrics(binary_sequence: Sequence[int]) -> RunStats:
    runs: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(binary_sequence, start=1):
        if value == 1 and start is None:
            start = idx
        if value == 0 and start is not None:
            runs.append((start, idx - start))
            start = None
    if start is not None:
        runs.append((start, len(binary_sequence) + 1 - start))

    if not runs:
        return RunStats(0.0, 0.0, 0.0)

    lengths = [length for _, length in runs]
    positions = [pos for pos, _ in runs]
    return RunStats(avg_length=mean(lengths), avg_count=float(len(runs)), avg_position=mean(positions))


def grouped_by_concurso(rows: Iterable[MatchRow]) -> Dict[str, List[MatchRow]]:
    grouped: Dict[str, List[MatchRow]] = {}
    for row in rows:
        grouped.setdefault(row.concurso, []).append(row)
    for concurso_rows in grouped.values():
        concurso_rows.sort(key=lambda r: r.jogo)
    return grouped


def dump_json(path: str, payload: Dict[str, object]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(level: int = logging.INFO, logfile: str = "output/debug.log") -> None:
    ensure_dir(str(Path(logfile).parent))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
