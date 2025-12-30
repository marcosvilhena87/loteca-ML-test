import numpy as np
import pandas as pd


def load_rateio(rateio_file: str) -> pd.DataFrame:
    """Load and normalize the rateio file, cleaning BOM and column names."""
    r = pd.read_csv(rateio_file, sep=None, engine="python")
    r.columns = [c.replace("\ufeff", "").strip() for c in r.columns]

    rename = {
        "Concurso": "Concurso",
        "Ganhadores 14 Acertos": "Ganhadores_14",
        "Rateio 14 Acertos": "Rateio_14",
        "Arrecadacao Total": "Arrecadacao_Total",
    }
    r = r.rename(columns=rename)

    r["Concurso"] = pd.to_numeric(r["Concurso"], errors="coerce").astype("Int64")
    r["Ganhadores_14"] = (
        pd.to_numeric(r["Ganhadores_14"], errors="coerce").fillna(0).astype(int)
    )
    r["Rateio_14"] = pd.to_numeric(r["Rateio_14"], errors="coerce").fillna(0.0)
    r["Arrecadacao_Total"] = pd.to_numeric(
        r["Arrecadacao_Total"], errors="coerce"
    )

    r["Log_Rateio_14"] = np.log1p(r["Rateio_14"])
    r["Acumulou_14"] = (r["Ganhadores_14"] == 0).astype(int)
    return r
