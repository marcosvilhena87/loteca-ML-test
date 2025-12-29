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
    for col in ['P(1)', 'P(X)', 'P(2)', 'Form_Diff_Last5', 'Is_Home', 'Home_Fav', 'Market_vs_Form']:
        assert col in df.columns
    assert 'Resultado' in df.columns
