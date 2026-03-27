import argparse
from statistics import mean, pstdev
from typing import Dict, List

from scripts.common import (
    MatchRow,
    grouped_by_concurso,
    normalize_row,
    read_csv_semicolon,
    runs_metrics,
    write_csv_semicolon,
)


def compute_targets(history: List[MatchRow]) -> Dict[str, Dict[str, float]]:
    grouped = grouped_by_concurso(history)
    targets: Dict[str, Dict[str, float]] = {}

    for slot_idx, slot_name in enumerate(["top1", "top2", "top3"], start=1):
        run_lengths = []
        run_counts = []
        run_positions = []

        for _, rows in grouped.items():
            sorted_rows = sorted(rows, key=lambda r: getattr(r, f"p_top{slot_idx}"), reverse=True)
            sequence = [getattr(r, f"top{slot_idx}_hit") for r in sorted_rows]
            stats = runs_metrics(sequence)
            run_lengths.append(stats.avg_length)
            run_counts.append(stats.avg_count)
            run_positions.append(stats.avg_position)

        targets[slot_name] = {
            "avg_length": mean(run_lengths) if run_lengths else 0.0,
            "avg_count": mean(run_counts) if run_counts else 0.0,
            "avg_position": mean(run_positions) if run_positions else 0.0,
            "avg_length_std": pstdev(run_lengths) if run_lengths else 0.0,
            "avg_count_std": pstdev(run_counts) if run_counts else 0.0,
            "avg_position_std": pstdev(run_positions) if run_positions else 0.0,
        }
    return targets


def run(input_path: str, output_path: str) -> None:
    raw_rows = read_csv_semicolon(input_path)
    normalized = [normalize_row(row) for row in raw_rows]
    targets = compute_targets(normalized)

    rows = []
    for row in normalized:
        rows.append(
            {
                "Concurso": row.concurso,
                "Jogo": row.jogo,
                "Mandante": row.mandante,
                "Visitante": row.visitante,
                "Data": row.data,
                "p(1)": f"{row.p1:.6f}",
                "p(x)": f"{row.px:.6f}",
                "p(2)": f"{row.p2:.6f}",
                "p(top1)": f"{row.p_top1:.6f}",
                "p(top2)": f"{row.p_top2:.6f}",
                "p(top3)": f"{row.p_top3:.6f}",
                "top1": row.top1_hit,
                "top2": row.top2_hit,
                "top3": row.top3_hit,
            }
        )

    write_csv_semicolon(
        output_path,
        rows,
        fieldnames=[
            "Concurso",
            "Jogo",
            "Mandante",
            "Visitante",
            "Data",
            "p(1)",
            "p(x)",
            "p(2)",
            "p(top1)",
            "p(top2)",
            "p(top3)",
            "top1",
            "top2",
            "top3",
        ],
    )

    print("Targets:")
    for slot, values in targets.items():
        print(slot, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessa histórico da Loteca.")
    parser.add_argument("--input", default="data/concursos_anteriores.csv")
    parser.add_argument("--output", default="output/preprocessed_history.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
