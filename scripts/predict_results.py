import logging
import os
import numpy as np
import pandas as pd
from joblib import load  # Para carregar os modelos previamente treinados

from scripts.preprocess_data import (
    FEATURE_COLUMNS,
    add_common_features,
    build_team_history,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _extract_ordered_probabilities(model, probabilities):
    """Reorder probability columns according to the class labels 1, X, 2."""

    expected_classes = ['1', 'X', '2']
    model_classes = list(model.classes_)
    missing_classes = set(expected_classes) - set(model_classes)
    if missing_classes:
        raise ValueError(f"Classes esperadas ausentes no modelo: {missing_classes}")

    prob_df = pd.DataFrame(probabilities, columns=model_classes)
    ordered = prob_df[expected_classes]
    return ordered


def predict(
    input_file,
    model_file,
    output_file,
    historical_file: str = "data/raw/concursos_anteriores.csv",
):
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
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Construindo histórico dos times a partir dos concursos anteriores...")
        if not os.path.exists(historical_file):
            raise FileNotFoundError(
                f"Arquivo de histórico não encontrado em '{historical_file}'"
            )

        historical_df = pd.read_csv(historical_file, delimiter=';', decimal='.')
        history = build_team_history(historical_df)

        logging.info("Gerando o mesmo conjunto de features do treinamento...")
        df_features = add_common_features(df, base_history=history)

        required_columns = FEATURE_COLUMNS
        
        # Carregando o modelo
        logging.info("Carregando modelo...")
        model = load(model_file)

        # Selecionando as features para predição
        logging.info("Preparando dados para predição...")
        X_future = df_features[required_columns]

        # Gerando as predições
        logging.info("Gerando predições...")
        raw_probabilities = model.predict_proba(X_future)
        ordered_probabilities = _extract_ordered_probabilities(model, raw_probabilities)
        predictions = model.predict(X_future)
        sorted_probs = np.sort(ordered_probabilities.to_numpy())[:, ::-1]

        # Adicionando as predições ao DataFrame
        logging.info("Adicionando predições ao DataFrame...")
        df_features['Probabilidade (1)'] = np.round(ordered_probabilities['1'].to_numpy(), 5)
        df_features['Probabilidade (X)'] = np.round(ordered_probabilities['X'].to_numpy(), 5)
        df_features['Probabilidade (2)'] = np.round(ordered_probabilities['2'].to_numpy(), 5)
        df_features['Seco'] = predictions
        df_features['model_top_margin'] = np.round(sorted_probs[:, 0] - sorted_probs[:, 1], 5)

        # Calculando a entropia com clipping para estabilidade numérica
        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        probabilities_clipped = np.clip(ordered_probabilities.to_numpy(), 1e-12, 1.0)
        df_features['Entropia'] = -np.sum(probabilities_clipped * np.log(probabilities_clipped), axis=1)

        # Identificar os 5 jogos mais incertos para aplicar os "duplos"
        jogos_duplos_idxs = df_features.nlargest(5, 'Entropia').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        # Gerar a coluna "Aposta"
        logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
        df_features['Aposta'] = df_features['Seco']  # Copia as apostas secas inicialmente

        # Escolhendo os "duplos" para os 5 jogos mais incertos
        duplo_opcoes = ['1', 'X', '2']
        prob_matrix = ordered_probabilities.to_numpy()
        for idx in jogos_duplos_idxs:
            mais_provaveis = prob_matrix[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
            df_features.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"

        # Salvando as predições no arquivo
        logging.info(f"Salvando predições no arquivo {output_file}...")
        df_features.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")
    
    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
