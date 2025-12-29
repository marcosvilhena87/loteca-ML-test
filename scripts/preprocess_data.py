import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

RATEIO_COLUMNS = {
    "Concurso": "Concurso",
    "Ganhadores 14 Acertos": "ganhadores_14",
    "Rateio 14 Acertos": "rateio_14",
}


def _normalize_rateio_data(rateio_file: str) -> pd.DataFrame:
    """Read and normalize the ``concurso_rateio.csv`` file.

    Parameters
    ----------
    rateio_file : str
        Path to the CSV containing prize information per ``Concurso``.

    Returns
    -------
    pd.DataFrame
        Normalized dataframe with engineered features ready for merge.
    """
    logging.info("Carregando dados de rateio...")
    rateio_df = pd.read_csv(rateio_file, delimiter=';', decimal='.')

    missing_cols = [col for col in RATEIO_COLUMNS if col not in rateio_df.columns]
    if missing_cols:
        raise KeyError(f"As seguintes colunas estão ausentes no rateio: {missing_cols}")

    # Renomear e garantir tipos
    rateio_df = rateio_df[list(RATEIO_COLUMNS.keys())].rename(columns=RATEIO_COLUMNS)
    rateio_df['Concurso'] = pd.to_numeric(rateio_df['Concurso'], errors='coerce').astype('Int64')
    rateio_df['ganhadores_14'] = pd.to_numeric(rateio_df['ganhadores_14'], errors='coerce').astype('Int64')
    rateio_df['rateio_14'] = pd.to_numeric(rateio_df['rateio_14'], errors='coerce').astype(float)

    # Feature engineering
    rateio_df['jackpot_14'] = rateio_df['ganhadores_14'] == 0
    rateio_df['log_rateio_14'] = np.log1p(rateio_df['rateio_14'])

    # Bins de rateio para análises/relatórios
    try:
        rateio_df['faixa_rateio'] = pd.qcut(rateio_df['log_rateio_14'], 4, labels=False, duplicates='drop')
    except ValueError:
        # Caso não haja variabilidade suficiente
        rateio_df['faixa_rateio'] = 0

    # Sequência de jackpots (rollover)
    rateio_df = rateio_df.sort_values('Concurso').reset_index(drop=True)
    streak = 0
    streaks = []
    for jackpot in rateio_df['jackpot_14']:
        streak = streak + 1 if jackpot else 0
        streaks.append(streak)
    rateio_df['rollover_streak'] = streaks

    return rateio_df


def process(input_file: str, output_file: str, rateio_file: Optional[str] = None):
    """Clean historical data and compute probabilities.

    Parameters
    ----------
    input_file : str
        CSV file with previous match results and odds.
    output_file : str
        Destination path for the processed CSV.

    Returns
    -------
    None
        The processed file is written to ``output_file``.
    """
    try:
        # Carregar os dados
        logging.info("Carregando dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        # Verificar se as colunas de odds estão presentes
        odds_columns = ['Odds 1', 'Odds X', 'Odds 2']
        missing_columns = [col for col in odds_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"As seguintes colunas necessárias estão ausentes: {missing_columns}")

        # Calcular probabilidades inversas a partir das odds
        logging.info("Calculando probabilidades baseadas nas odds...")
        df['P(1)'] = 1 / df['Odds 1']
        df['P(X)'] = 1 / df['Odds X']
        df['P(2)'] = 1 / df['Odds 2']

        # Normalizar as probabilidades para somarem 1 e calcular overround
        logging.info("Normalizando probabilidades...")
        prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
        df['overround'] = prob_sum
        df['P(1)'] /= prob_sum
        df['P(X)'] /= prob_sum
        df['P(2)'] /= prob_sum

        # Garantir a coluna 'Resultado'
        logging.info("Determinando os resultados reais dos jogos...")
        if all(col in df.columns for col in ['[1]', '[x]', '[2]']):
            df['Resultado'] = df.apply(lambda row: '1' if row['[1]'] == 1 else
                                                   'X' if row['[x]'] == 1 else
                                                   '2' if row['[2]'] == 1 else None, axis=1)
        else:
            raise KeyError("As colunas '[1]', '[x]' e '[2]' são necessárias para calcular o resultado.")

        # Remover linhas inválidas
        logging.info("Removendo linhas inválidas...")
        df = df.dropna(subset=['Resultado', 'P(1)', 'P(X)', 'P(2)'])

        # Features derivadas de probabilidade
        logging.info("Calculando features derivadas das probabilidades...")
        probs = df[['P(1)', 'P(X)', 'P(2)']].to_numpy()
        df['pmax'] = probs.max(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        df['gap12'] = sorted_probs[:, 2] - sorted_probs[:, 1]
        epsilon = 1e-10
        df['entropy'] = -np.sum((probs + epsilon) * np.log(probs + epsilon), axis=1)

        # Integração com rateio por Concurso
        if rateio_file is not None:
            if 'Concurso' not in df.columns:
                raise KeyError("Coluna 'Concurso' é necessária para integrar o rateio.")

            rateio_df = _normalize_rateio_data(rateio_file)
            merged_before = len(df)
            df = df.merge(rateio_df, on='Concurso', how='left')
            matched = df['rateio_14'].notna().sum()
            logging.info(
                "Merge de rateio concluído: %s linhas originais, %s linhas com rateio." % (merged_before, matched)
            )
            missing = merged_before - matched
            if missing:
                logging.warning(
                    "%s concursos ficaram sem informações de rateio após o merge (campos permanecerão NaN).",
                    missing,
                )

        # Salvar o arquivo processado
        logging.info(f"Salvando o arquivo processado em {output_file}...")
        df.to_csv(output_file, index=False, sep=';', decimal='.')
        logging.info("Processamento concluído com sucesso!")
    
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
