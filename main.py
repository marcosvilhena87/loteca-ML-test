import logging

from scripts.predict_results import predict
from scripts.train_model import train_model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_model(
        history_path="data/concursos_anteriores.csv",
        model_path="models/model.json",
        preprocessed_path="output/preprocessed_history.csv",
    )
    predict(
        next_path="data/proximo_concurso.csv",
        model_path="models/model.json",
        output_path="output/predictions.csv",
    )


if __name__ == "__main__":
    main()
