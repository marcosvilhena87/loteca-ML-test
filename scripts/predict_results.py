import logging
import pandas as pd
import numpy as np
from joblib import load  # Para carregar os modelos previamente treinados

from .feature_engineering import (
    CLASSES,
    MODEL_FEATURES,
    PROB_COLUMNS,
    add_domain_features,
    compute_probabilities,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def _reorder_probabilities(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return probability matrix with columns aligned to ``CLASSES`` order."""
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    missing = [cls for cls in CLASSES if cls not in class_to_index]
    if missing:
        raise ValueError(
            f"Classes esperadas ausentes do modelo: {missing}. Classes disponíveis: {list(classes)}"
        )
    return np.column_stack([proba[:, class_to_index[cls]] for cls in CLASSES])


def _pick_indices(df: pd.DataFrame, filters, sort_col: str, target: int, exclude=None):
    """Pick indices using successive filters until ``target`` rows are found."""

    available = df if exclude is None else df.drop(index=exclude)

    for i, filter_step in enumerate(filters):
        subset = filter_step(available)
        if len(subset) >= target or i == len(filters) - 1:
            return subset.nlargest(target, sort_col).index
    return pd.Index([])


def predict(input_file, model_file, scaler_file, output_file):
    """Generate predictions for future games and write them to CSV."""
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Calculando probabilidades das partidas...")
        df = compute_probabilities(df)

        logging.info("Criando features específicas da Loteca...")
        df = add_domain_features(df)

        logging.info("Carregando modelo...")
        model = load(model_file)

        logging.info(f"Ordem das classes do modelo: {list(model.classes_)}")

        logging.info("Preparando dados para predição...")
        X_future = df[MODEL_FEATURES]

        logging.info("Gerando predições...")
        probabilities = _reorder_probabilities(model.predict_proba(X_future), model.classes_)

        logging.info("Adicionando predições ao DataFrame com mapeamento correto das classes...")
        df['Probabilidade (1)'] = np.round(probabilities[:, 0], 5)
        df['Probabilidade (X)'] = np.round(probabilities[:, 1], 5)
        df['Probabilidade (2)'] = np.round(probabilities[:, 2], 5)

        market_top = df[PROB_COLUMNS].to_numpy().argmax(axis=1)
        df['Seco_Mercado'] = [CLASSES[idx] for idx in market_top]

        predictions_mapped = [CLASSES[idx] for idx in probabilities.argmax(axis=1)]
        df['Seco_Modelo'] = predictions_mapped

        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon

        logging.info("Calculando entropia para determinar os jogos mais incertos...")
        df['Entropia'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)
        df['Pmax_Modelo'] = probabilities.max(axis=1)
        df['Score_Duplo'] = df['Entropia'] * (1 - df['Pmax_Modelo'])

        triplo_filters = [
            lambda d: d[(d['Pmax_Modelo'] <= 0.55) & (d['Probabilidade (X)'] >= 0.20)],
            lambda d: d[(d['Pmax_Modelo'] <= 0.65)],
            lambda d: d,
        ]
        jogos_triplos_idxs = _pick_indices(df, triplo_filters, 'Entropia', 3)
        logging.info("Gerando a coluna de aposta com 6 secos, 5 duplos e 3 triplos...")

        candidatos_duplo_filters = [
            lambda d: d[(d['Pmax_Modelo'] <= 0.60) & (d['Probabilidade (X)'] >= 0.22)],
            lambda d: d[(d['Pmax_Modelo'] <= 0.70) & (d['Probabilidade (X)'] >= 0.18)],
            lambda d: d,
        ]

        jogos_duplos_idxs = _pick_indices(
            df,
            candidatos_duplo_filters,
            'Score_Duplo',
            5,
            exclude=jogos_triplos_idxs,
        )
        logging.info(f"Índices escolhidos para triplos: {jogos_triplos_idxs.tolist()}")
        logging.info(f"Índices escolhidos para duplos: {jogos_duplos_idxs.tolist()}")

        df['Aposta'] = df['Seco_Mercado']

        df['Duplo_Modelo'] = ""
        df['Triplo_Modelo'] = ""

        for idx in jogos_triplos_idxs:
            df.loc[idx, 'Aposta'] = '1, X, 2'
            df.loc[idx, 'Triplo_Modelo'] = '1, X, 2'

        duplo_opcoes = ['1', 'X', '2']
        for idx in jogos_duplos_idxs:
            mais_provaveis = probabilities[idx].argsort()[-2:][::-1]
            duplos_ordenados = sorted(
                (duplo_opcoes[mais_provaveis[0]], duplo_opcoes[mais_provaveis[1]]),
                key=['1', 'X', '2'].index,
            )
            duplo = ", ".join(duplos_ordenados)
            df.loc[idx, 'Aposta'] = duplo
            df.loc[idx, 'Duplo_Modelo'] = duplo

        logging.info(f"Salvando predições no arquivo {output_file}...")
        df.to_csv(output_file, sep=';', index=False)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
