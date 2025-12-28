import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import process, train


def test_train_writes_metrics_file(tmp_path):
    raw_file = "data/raw/concursos_anteriores.csv"
    processed_file = tmp_path / "processed.csv"
    metrics_file = tmp_path / "metrics.json"

    process(raw_file, processed_file)
    train(processed_file, metrics_file)

    assert metrics_file.exists()
