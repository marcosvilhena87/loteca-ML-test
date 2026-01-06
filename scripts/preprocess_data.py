import logging
import pandas as pd

from scripts.features import FEATURE_COLUMNS, add_engineered_features

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _compute_result(row):
    if row.get('[1]') == 1:
        return '1'
    if row.get('[x]') == 1:
        return 'X'
    if row.get('[2]') == 1:
        return '2'
    return None


def process(input_file, output_file):
    """Limpa e enriquece o histórico com novas features."""
    try:
        logging.info("Carregando dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Gerando features derivadas...")
        df = add_engineered_features(df)

        logging.info("Determinando o resultado real de cada jogo...")
        if all(col in df.columns for col in ['[1]', '[x]', '[2]']):
            df['Resultado'] = df.apply(_compute_result, axis=1)
        else:
            raise KeyError("As colunas '[1]', '[x]' e '[2]' são necessárias para calcular o resultado.")

        logging.info("Removendo linhas inválidas...")
        df = df.dropna(subset=['Resultado'] + FEATURE_COLUMNS)

        logging.info(f"Salvando o arquivo processado em {output_file}...")
        df.to_csv(output_file, index=False, sep=';', decimal='.')
        logging.info("Processamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
