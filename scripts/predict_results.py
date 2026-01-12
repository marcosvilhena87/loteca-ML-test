import logging
import pandas as pd
import numpy as np
from joblib import load  # Para carregar os modelos previamente treinados

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def predict(input_file, model_file, scaler_file, output_file):
    """Generate predictions for future games and write them to CSV.

    Parameters
    ----------
    input_file : str
        CSV file with upcoming games and odds or probabilities.
    model_file : str
        Path of the trained model to load.
    scaler_file : str
        Path of the scaler used during training.
    output_file : str
        Destination path for the predictions CSV.

    Returns
    -------
    None
        The predictions are saved to ``output_file``.
    """
    try:
        # Carregando os dados dos jogos futuros
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        # Verificando se as colunas de probabilidades estão presentes
        required_prob_cols = ['P(1)', 'P(X)', 'P(2)']
        if not all(col in df.columns for col in required_prob_cols):
            # Caso as colunas de probabilidade não existam, calculá-las a partir das odds
            odds_cols = ['Odds_1', 'Odds_X', 'Odds_2']
            if all(col in df.columns for col in odds_cols):
                logging.info("Colunas de probabilidade ausentes. Calculando a partir das odds...")
                df['P(1)'] = 1 / df['Odds_1']
                df['P(X)'] = 1 / df['Odds_X']
                df['P(2)'] = 1 / df['Odds_2']

                prob_sum = df['P(1)'] + df['P(X)'] + df['P(2)']
                df['P(1)'] /= prob_sum
                df['P(X)'] /= prob_sum
                df['P(2)'] /= prob_sum
            else:
                raise ValueError(
                    f"As colunas de odds {odds_cols} são necessárias para calcular as probabilidades no arquivo {input_file}."
                )

        # Reconfirmando que as colunas de probabilidade agora estão presentes
        required_columns = ['P(1)', 'P(X)', 'P(2)']
        
        # Carregando o modelo e o scaler
        logging.info("Carregando modelo e scaler...")
        model = load(model_file)
        scaler = load(scaler_file)

        # Selecionando as features para predição e escalando
        logging.info("Preparando dados para predição...")
        X_future = df[required_columns]
        X_future_scaled = scaler.transform(X_future)

        # Gerando as predições
        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future_scaled)
        predictions = model.predict(X_future_scaled)

        classes = list(model.classes_)
        proba_df = pd.DataFrame(probabilities, columns=classes, index=df.index)
        for c in ['1', 'X', '2']:
            if c not in proba_df.columns:
                proba_df[c] = 0.0
        proba_df = proba_df[['1', 'X', '2']]
        ordered_probabilities = proba_df.to_numpy()

        # Adicionando as predições ao DataFrame
        logging.info("Adicionando predições ao DataFrame...")
        df['Probabilidade (1)'] = np.round(proba_df['1'], 5)
        df['Probabilidade (X)'] = np.round(proba_df['X'], 5)
        df['Probabilidade (2)'] = np.round(proba_df['2'], 5)
        df['Secos'] = predictions

        # Adicionando um valor pequeno para evitar problemas com log(0)
        epsilon = 1e-10
        adjusted_probabilities = ordered_probabilities + epsilon

        # Calculando a entropia com as probabilidades ajustadas
        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        # Identificar os 5 jogos mais incertos para aplicar os "duplos"
        jogos_duplos_idxs = df.nlargest(5, 'Entropia').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        # Gerar a coluna "Aposta"
        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Secos']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        duplo_opcoes = ['1', 'X', '2']
        for idx in jogos_duplos_idxs:
            mais_provaveis = ordered_probabilities[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
            df.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"

        # Salvando as predições no arquivo
        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")
    
    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
