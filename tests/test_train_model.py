import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import process, train


def test_train_outputs_model_files(tmp_path):
    raw_file = "data/raw/concursos_anteriores.csv"
    processed_file = tmp_path / "processed.csv"
    model_file = tmp_path / "model.pkl"

    process(raw_file, processed_file)
    train(processed_file, model_file, None)

    assert model_file.exists()
