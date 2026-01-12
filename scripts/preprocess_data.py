import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def process(input_file, output_file):
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
        odds_columns = ['Odds_1', 'Odds_X', 'Odds_2']
        missing_columns = [col for col in odds_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"As seguintes colunas necessárias estão ausentes: {missing_columns}")

        # Calcular probabilidades inversas a partir das odds
        logging.info("Calculando probabilidades baseadas nas odds...")
        df['P(1)'] = 1 / df['Odds_1']
        df['P(X)'] = 1 / df['Odds_X']
        df['P(2)'] = 1 / df['Odds_2']

        # Normalizar as probabilidades para somarem 1
        logging.info("Normalizando probabilidades...")
        prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
        df['P(1)'] /= prob_sum
        df['P(X)'] /= prob_sum
        df['P(2)'] /= prob_sum

        # Features derivadas simples para enriquecer o contexto
        logging.info("Calculando features derivadas das probabilidades...")
        eps = 1e-12
        df['log_odds_1_2'] = np.log((df['P(1)'] + eps) / (df['P(2)'] + eps))
        df['log_odds_1_X'] = np.log((df['P(1)'] + eps) / (df['P(X)'] + eps))
        df['log_odds_X_2'] = np.log((df['P(X)'] + eps) / (df['P(2)'] + eps))
        df['p1_minus_p2'] = df['P(1)'] - df['P(2)']
        df['pX_minus_max'] = df['P(X)'] - df[['P(1)', 'P(2)']].max(axis=1)
        df['fav_strength'] = df[['P(1)', 'P(2)']].max(axis=1) - df[['P(1)', 'P(2)']].min(axis=1)
        probs = df[['P(1)', 'P(X)', 'P(2)']].to_numpy()
        sorted_probs = np.sort(probs, axis=1)
        df['confidence_gap'] = sorted_probs[:, -1] - sorted_probs[:, -2]
        df['entropy_market'] = -np.sum(
            probs * np.log(probs + eps),
            axis=1,
        )

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
        df = df.dropna(
            subset=[
                'Resultado',
                'P(1)',
                'P(X)',
                'P(2)',
                'log_odds_1_2',
                'log_odds_1_X',
                'log_odds_X_2',
                'p1_minus_p2',
                'pX_minus_max',
                'fav_strength',
                'confidence_gap',
                'entropy_market',
            ]
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
