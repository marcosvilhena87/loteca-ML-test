import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import process


def test_preprocess_creates_probability_columns(tmp_path):
    input_file = "data/raw/concursos_anteriores.csv"
    output_file = tmp_path / "processed.csv"
    process(input_file, output_file)

    df = pd.read_csv(output_file, delimiter=';', decimal='.')
    expected_columns = [
        'P(1)', 'P(X)', 'P(2)',
        'Pmax', 'Psecond', 'Gap', 'Entropy',
        'LogOdds_1', 'LogOdds_X', 'LogOdds_2',
        'DrawBias', 'DrawEntropyInteraction', 'DrawGapInteraction'
    ]
    for col in expected_columns:
        assert col in df.columns
