import logging
import pandas as pd

from .feature_engineering import (
    MODEL_FEATURES,
    add_domain_features,
    compute_probabilities,
)
from .rateio_utils import load_rateio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def process(input_file, output_file, rateio_file=None):
    """Clean historical data, compute probabilities and features, and optionally merge rateio."""
    try:
        logging.info("Carregando dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Calculando probabilidades baseadas nas odds...")
        df = compute_probabilities(df)

        logging.info("Determinando os resultados reais dos jogos...")
        if all(col in df.columns for col in ['[1]', '[x]', '[2]']):
            df['Resultado'] = df.apply(
                lambda row: '1' if row['[1]'] == 1 else
                'X' if row['[x]'] == 1 else
                '2' if row['[2]'] == 1 else None,
                axis=1
            )
        else:
            raise KeyError("As colunas '[1]', '[x]' e '[2]' são necessárias para calcular o resultado.")

        logging.info("Criando features específicas da Loteca...")
        df = add_domain_features(df)

        if rateio_file and "Concurso" in df.columns:
            logging.info("Carregando e unindo dados de rateio por concurso...")
            rateio = load_rateio(rateio_file)
            df["Concurso"] = pd.to_numeric(df["Concurso"], errors="coerce").astype("Int64")
            df = df.merge(rateio, on="Concurso", how="left")

        logging.info("Removendo linhas inválidas...")
        df = df.dropna(subset=['Resultado'] + MODEL_FEATURES)

        logging.info(f"Salvando o arquivo processado em {output_file}...")
        df.to_csv(output_file, index=False, sep=';', decimal='.')
        logging.info("Processamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
