import logging
import pandas as pd
import numpy as np
from joblib import load  # Para carregar os modelos previamente treinados

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def predict(input_file, model_file, output_file):
    """Generate predictions for future games and write them to CSV.

    Parameters
    ----------
    input_file : str
        CSV file with upcoming games and odds or probabilities.
    model_file : str
        Path of the trained model to load.
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
            odds_cols = ['Odds 1', 'Odds X', 'Odds 2']
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

        # Reconfirmando que as colunas de probabilidade agora estão presentes
        required_columns = ['P(1)', 'P(X)', 'P(2)']
        
        # Carregando o modelo
        logging.info("Carregando modelo...")
        model = load(model_file)

        # Selecionando as features para predição
        logging.info("Preparando dados para predição...")
        X_future = df[required_columns]

        # Gerando as predições
        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future)
        predictions = model.predict(X_future)

        # Garantindo mapeamento correto classe -> coluna
        classes = model.classes_
        class_indices = {label: idx for idx, label in enumerate(classes)}
        try:
            idx_1 = class_indices['1']
            idx_X = class_indices['X']
            idx_2 = class_indices['2']
        except KeyError as missing:
            raise ValueError(f"Classe esperada ausente no modelo: {missing}")

        # Adicionando as predições ao DataFrame
        logging.info("Adicionando predições ao DataFrame...")
        df['Probabilidade (1)'] = np.round(probabilities[:, idx_1], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, idx_X], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, idx_2], 5)
        df['Seco'] = predictions

        # Adicionando um valor pequeno para evitar problemas com log(0)
        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon

        # Calculando a entropia com as probabilidades ajustadas
        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)

        # Identificar os 5 jogos mais incertos para aplicar os "duplos"
        jogos_duplos_idxs = df.nlargest(5, 'Entropia').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        # Gerar a coluna "Aposta"
        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df['Aposta'] = df['Seco']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        for idx in jogos_duplos_idxs:
            mais_provaveis_idxs = probabilities[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
            mais_provaveis_labels = [classes[i] for i in mais_provaveis_idxs]
            df.loc[idx, 'Aposta'] = ", ".join(mais_provaveis_labels)

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
