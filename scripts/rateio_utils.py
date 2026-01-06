"""Utilitários para leitura e normalização dos arquivos de rateio."""

import pandas as pd
import numpy as np


def load_rateio(file_path: str) -> pd.DataFrame:
    """Carrega o CSV de rateio, normalizando nomes e adicionando métricas úteis.

    - Renomeia colunas para padronizar nomenclatura.
    - Remove BOM e espaços supérfluos.
    - Calcula log do rateio de 14 acertos para evitar escala explosiva.
    - Marca concursos acumulados (sem ganhadores de 14 acertos).
    """

    df = pd.read_csv(file_path, delimiter=';', decimal='.')
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    rename_map = {
        'Ganhadores 14 Acertos': 'Ganhadores_14',
        'Rateio 14 Acertos': 'Rateio_14',
        'Arrecadacao Total': 'Arrecadacao_Total',
    }
    df = df.rename(columns=rename_map)

    required = ['Concurso', 'Ganhadores_14', 'Rateio_14', 'Arrecadacao_Total']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes no rateio: {missing}")

    df['Log_Rateio_14'] = np.log1p(df['Rateio_14'])
    df['Acumulou_14'] = (df['Ganhadores_14'] == 0).astype(int)

    return df[required + ['Log_Rateio_14', 'Acumulou_14']]
