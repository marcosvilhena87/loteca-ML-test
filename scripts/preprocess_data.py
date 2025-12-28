import logging
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
        odds_columns = ['Odds 1', 'Odds X', 'Odds 2']
        missing_columns = [col for col in odds_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"As seguintes colunas necessárias estão ausentes: {missing_columns}")

        # Calcular probabilidades inversas a partir das odds
        logging.info("Calculando probabilidades baseadas nas odds...")
        df['P(1)'] = 1 / df['Odds 1']
        df['P(X)'] = 1 / df['Odds X']
        df['P(2)'] = 1 / df['Odds 2']

        # Normalizar as probabilidades para somarem 1
        logging.info("Normalizando probabilidades...")
        prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
        df['P(1)'] /= prob_sum
        df['P(X)'] /= prob_sum
        df['P(2)'] /= prob_sum

        # Garantir a coluna 'Resultado'
        logging.info("Determinando os resultados reais dos jogos...")
        resultado_cols = ['[1]', '[x]', '[2]']
        if all(col in df.columns for col in resultado_cols):
            escolha_unica = df[resultado_cols].sum(axis=1) == 1
            linhas_invalidas = (~escolha_unica).sum()

            if linhas_invalidas > 0:
                logging.warning(
                    f"{linhas_invalidas} linhas removidas por terem múltiplas ou nenhuma marcação em {resultado_cols}."
                )

            df = df[escolha_unica]
            df['Resultado'] = df.apply(lambda row: '1' if row['[1]'] == 1 else
                                                   'X' if row['[x]'] == 1 else
                                                   '2', axis=1)
        else:
            raise KeyError("As colunas '[1]', '[x]' e '[2]' são necessárias para calcular o resultado.")

        # Remover linhas inválidas
        logging.info("Removendo linhas inválidas...")
        df = df.dropna(subset=['Resultado', 'P(1)', 'P(X)', 'P(2)'])

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
