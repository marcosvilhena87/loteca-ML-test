import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import process, train, predict


def test_predict_results_output_columns(tmp_path):
    raw_train = "data/raw/concursos_anteriores.csv"
    processed_file = tmp_path / "processed.csv"
    model_file = tmp_path / "model.pkl"
    predictions_file = tmp_path / "predictions.csv"
    rateio_file = "data/raw/concurso_rateio.csv"

    process(raw_train, processed_file, rateio_file=rateio_file)
    train(processed_file, model_file)
    predict("data/raw/proximo_concurso.csv", model_file, predictions_file)

    df = pd.read_csv(predictions_file, delimiter=';')
    expected = [
        'Probabilidade (1)',
        'Probabilidade (X)',
        'Probabilidade (2)',
        'Seco',
        'Entropia',
        'Aposta'
    ]
    for col in expected:
        assert col in df.columns
