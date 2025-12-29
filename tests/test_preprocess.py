import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import process


def test_preprocess_creates_probability_columns(tmp_path):
    input_file = "data/raw/concursos_anteriores.csv"
    output_file = tmp_path / "processed.csv"
    rateio_file = "data/raw/concurso_rateio.csv"
    process(input_file, output_file, rateio_file=rateio_file)

    df = pd.read_csv(output_file, delimiter=';', decimal='.')
    assert 'P(1)' in df.columns
    assert 'P(X)' in df.columns
    assert 'P(2)' in df.columns
    assert 'rateio_14' in df.columns
    assert 'log_rateio_14' in df.columns
