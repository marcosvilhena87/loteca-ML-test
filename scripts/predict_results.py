import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def predict(input_file, output_file):
    """Generate baseline predictions for future games and write them to CSV.

    Parameters
    ----------
    input_file : str
        CSV file with upcoming games and odds or probabilities.
    output_file : str
        Destination path for the predictions CSV.

    Returns
    -------
    None
        The predictions are saved to ``output_file``.
    """
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        required_prob_cols = ['P(1)', 'P(X)', 'P(2)']
        odds_cols = ['Odds 1', 'Odds X', 'Odds 2']

        if not all(col in df.columns for col in required_prob_cols):
            if all(col in df.columns for col in odds_cols):
                logging.info("Colunas de probabilidade ausentes. Calculando a partir das odds...")
                df['P(1)'] = 1 / df['Odds 1']
                df['P(X)'] = 1 / df['Odds X']
                df['P(2)'] = 1 / df['Odds 2']

                prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
                df['P(1)'] /= prob_sum
                df['P(X)'] /= prob_sum
                df['P(2)'] /= prob_sum
            else:
                raise ValueError(
                    f"As colunas de odds {odds_cols} são necessárias para calcular as probabilidades no arquivo {input_file}."
                )

        logging.info("Usando baseline (argmax das probabilidades implícitas) para gerar predições...")
        prob_array = df[['P(1)', 'P(X)', 'P(2)']].to_numpy()
        classes = np.array(['1', 'X', '2'])

        df['Probabilidade (1)'] = np.round(prob_array[:, 0], 5)
        df['Probabilidade (X)'] = np.round(prob_array[:, 1], 5)
        df['Probabilidade (2)'] = np.round(prob_array[:, 2], 5)
        df['Seco'] = classes[prob_array.argmax(axis=1)]

        epsilon = 1e-10
        adjusted_probabilities = prob_array + epsilon

        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        jogos_duplos_idxs = df.nlargest(5, 'Entropia').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Seco']

        for idx in jogos_duplos_idxs:
            mais_provaveis = prob_array[idx].argsort()[-2:][::-1]
            opcoes_duplas = classes[mais_provaveis]
            df.loc[idx, 'Aposta'] = ", ".join(opcoes_duplas)

        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
