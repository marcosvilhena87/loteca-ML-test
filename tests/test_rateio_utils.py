import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rateio_utils import load_rateio


def test_load_rateio_normalizes_columns(tmp_path):
    rateio_content = """\ufeffConcurso;Ganhadores 14 Acertos;Rateio 14 Acertos;Arrecadacao Total
1;0;1000000;2000000
2;5;5000;1500000
"""
    rateio_file = tmp_path / "rateio.csv"
    rateio_file.write_text(rateio_content)

    df = load_rateio(rateio_file)

    assert list(df.columns) == [
        "Concurso",
        "Ganhadores_14",
        "Rateio_14",
        "Arrecadacao_Total",
        "Log_Rateio_14",
        "Acumulou_14",
    ]
    assert df.loc[0, "Concurso"] == 1
    assert df.loc[0, "Acumulou_14"] == 1
    assert df.loc[1, "Ganhadores_14"] == 5
