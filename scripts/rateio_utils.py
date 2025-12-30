"""Utilities for parsing Loteca rateio (prize) files."""
import numpy as np
import pandas as pd


def load_rateio(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', decimal='.', encoding='utf-8-sig')
    rename_map = {
        'Concurso': 'Concurso',
        'Ganhadores 14 Acertos': 'Ganhadores_14',
        'Rateio 14 Acertos': 'Rateio_14',
        'Arrecadacao Total': 'Arrecadacao_Total',
    }
    df = df.rename(columns=rename_map)
    df['Ganhadores_14'] = pd.to_numeric(df['Ganhadores_14'], errors='coerce').fillna(0).astype(int)
    df['Rateio_14'] = pd.to_numeric(df['Rateio_14'], errors='coerce')
    df['Arrecadacao_Total'] = pd.to_numeric(df['Arrecadacao_Total'], errors='coerce')
    df['Log_Rateio_14'] = df['Rateio_14'].replace(0, pd.NA).apply(lambda x: np.log(x) if pd.notna(x) else np.nan)
    df['Acumulou_14'] = (df['Ganhadores_14'] == 0).astype(int)
    return df[['Concurso', 'Ganhadores_14', 'Rateio_14', 'Arrecadacao_Total', 'Log_Rateio_14', 'Acumulou_14']]
