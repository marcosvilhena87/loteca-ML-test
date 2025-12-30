import logging
import pandas as pd

from .features import (RatingEngine, compute_expert_differences,
                       compute_implied_probabilities, enrich_features)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def process(input_file, output_file):
    """Clean historical data, compute expert signals and save enriched dataset."""
    try:
        logging.info("Carregando dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        odds_columns = ['Odds 1', 'Odds X', 'Odds 2']
        missing_columns = [col for col in odds_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"As seguintes colunas necessárias estão ausentes: {missing_columns}")

        logging.info("Calculando probabilidades baseadas nas odds e margem do mercado...")
        df = compute_implied_probabilities(df)
        df['P(1)'] = df['P_market(1)']
        df['P(X)'] = df['P_market(X)']
        df['P(2)'] = df['P_market(2)']

        logging.info("Determinando os resultados reais dos jogos...")
        if all(col in df.columns for col in ['[1]', '[x]', '[2]']):
            df['Resultado'] = df.apply(lambda row: '1' if row['[1]'] == 1 else
                                                  'X' if row['[x]'] == 1 else
                                                  '2' if row['[2]'] == 1 else None, axis=1)
        else:
            raise KeyError("As colunas '[1]', '[x]' e '[2]' são necessárias para calcular o resultado.")

        logging.info("Calculando features de Elo e Poisson (memória longa + empate)...")
        engine = RatingEngine()
        df = enrich_features(df, engine, update_results=True)

        logging.info("Criando deltas explícitos vs. mercado (Elo/Poisson)...")
        df = compute_expert_differences(df)

        logging.info(f"Salvando o arquivo processado em {output_file}...")
        df.to_csv(output_file, index=False, sep=';', decimal='.')
        logging.info("Processamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except KeyError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
