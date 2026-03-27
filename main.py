import argparse
import logging

from scripts.common import setup_logging
from scripts.predict_results import predict
from scripts.train_model import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline completo Loteca (treino + predição)")
    parser.add_argument("--history", default="data/concursos_anteriores.csv")
    parser.add_argument("--next", dest="next_path", default="data/proximo_concurso.csv")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--output", default="output/predictions.csv")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.info("Iniciando pipeline Loteca")
    train(args.history, args.model)
    predict(args.next_path, args.model, args.output)
    logging.info("Pipeline finalizado. Saída em %s", args.output)


if __name__ == "__main__":
    main()
