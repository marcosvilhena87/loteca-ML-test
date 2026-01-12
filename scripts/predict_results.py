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
        missing_prob_columns = [col for col in required_columns if col not in df.columns]
        if missing_prob_columns:
            raise ValueError(
                f"As colunas de probabilidade {missing_prob_columns} são necessárias no arquivo {input_file}."
            )

        # Garantir features derivadas iguais às usadas no treino
        logging.info("Calculando features derivadas para compatibilidade com o modelo...")
        eps = 1e-12
        df['log_odds_1_2'] = np.log((df['P(1)'] + eps) / (df['P(2)'] + eps))
        df['p1_minus_p2'] = df['P(1)'] - df['P(2)']
        probs = df[required_columns].to_numpy()
        sorted_probs = np.sort(probs, axis=1)
        df['confidence_gap'] = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # Carregando o modelo
        logging.info("Carregando modelo...")
        model = load(model_file)

        # Selecionando as features para predição
        logging.info("Preparando dados para predição...")
        feature_columns = [
            'P(1)',
            'P(X)',
            'P(2)',
            'log_odds_1_2',
            'p1_minus_p2',
            'confidence_gap',
        ]
        X_future = df[feature_columns]

        # Gerando as predições
        logging.info("Gerando predições...")
        probabilities = model.predict_proba(X_future)
        predictions = model.predict(X_future)

        # Adicionando as predições ao DataFrame
        logging.info("Adicionando predições ao DataFrame...")
        classes = [str(cls) for cls in model.classes_]
        cls_to_col = {cls: idx for idx, cls in enumerate(classes)}
        df['Probabilidade (1)'] = np.round(probabilities[:, cls_to_col['1']], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, cls_to_col['X']], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, cls_to_col['2']], 5)
        df['Secos'] = predictions

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
        df['Aposta'] = df['Secos']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        duplo_opcoes = classes
        for idx in jogos_duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
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
