import argparse
import logging
from collections import defaultdict
from statistics import mean

from scripts.common import detect_hit_rank, enrich_probabilities, run_features, write_csv_semicolon, read_csv_semicolon


def preprocess(history_path: str, output_path: str) -> dict:
    rows = read_csv_semicolon(history_path)
    logging.info("Carregados %d jogos do histórico", len(rows))

    by_concurso = defaultdict(list)
    processed_rows = []

    for row in rows:
        enriched = enrich_probabilities(row)
        hit_rank = detect_hit_rank(row, enriched)
        concurso = row["Concurso"]

        out = dict(row)
        out["pred_top1"] = enriched["_top_symbols"]["top1"]
        out["pred_top2"] = enriched["_top_symbols"]["top2"]
        out["pred_top3"] = enriched["_top_symbols"]["top3"]
        out["hit_rank"] = str(hit_rank)
        out["p_top1"] = f"{enriched['_top_probs']['top1']:.6f}"
        out["p_top2"] = f"{enriched['_top_probs']['top2']:.6f}"
        out["p_top3"] = f"{enriched['_top_probs']['top3']:.6f}"
        processed_rows.append(out)

        by_concurso[concurso].append((row, enriched, hit_rank))

    target_metrics = {}
    for rank_name, rank_idx in (("top1", 1), ("top2", 2), ("top3", 3)):
        per_concurso = []
        for jogos in by_concurso.values():
            ordered = sorted(jogos, key=lambda x: x[1]["_top_probs"][rank_name], reverse=True)
            binary = [1 if item[2] == rank_idx else 0 for item in ordered]
            per_concurso.append(run_features(binary))

        target_metrics[rank_name] = {
            "avg_run_length": mean(m["avg_run_length"] for m in per_concurso),
            "avg_run_count": mean(m["avg_run_count"] for m in per_concurso),
            "avg_run_position": mean(m["avg_run_position"] for m in per_concurso),
        }

    fieldnames = list(processed_rows[0].keys()) if processed_rows else []
    write_csv_semicolon(output_path, fieldnames, processed_rows)
    logging.info("Pré-processamento salvo em %s", output_path)

    return {
        "n_rows": len(processed_rows),
        "n_concursos": len(by_concurso),
        "target_metrics": target_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/concursos_anteriores.csv")
    parser.add_argument("--output", default="output/preprocessed_history.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    summary = preprocess(args.history, args.output)
    logging.info("Resumo: %s", summary)


if __name__ == "__main__":
    main()
