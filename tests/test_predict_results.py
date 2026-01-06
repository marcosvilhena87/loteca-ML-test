import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import process, train, predict


def test_predict_results_output_columns(tmp_path):
    raw_train = "data/raw/concursos_anteriores.csv"
    processed_file = tmp_path / "processed.csv"
    model_file = tmp_path / "model.pkl"
    scaler_file = tmp_path / "scaler.pkl"
    predictions_file = tmp_path / "predictions.csv"

    process(raw_train, processed_file)
    train(processed_file, model_file, scaler_file)
    predict("data/raw/proximo_concurso.csv", model_file, scaler_file, predictions_file)

    df = pd.read_csv(predictions_file, delimiter=';')
    expected = {
        'Probabilidade (1)',
        'Probabilidade (X)',
        'Probabilidade (2)',
        'Secos',
        'Seco',
        'Entropia',
        'p_seco',
        'risk',
        'score_duplo_tripo',
        'Aposta'
    }
    assert expected.issubset(set(df.columns))
