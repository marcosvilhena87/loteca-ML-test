import logging
import os
from typing import Optional

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


def _load_historical_dataframe(
    historical_file: str,
    historical_df: Optional[pd.DataFrame] = None,
    history_max_concurso: Optional[int] = None,
) -> pd.DataFrame:
    if historical_df is None:
        historical_df = pd.read_csv(historical_file, delimiter=";", decimal=".")

    if history_max_concurso is None:
        return historical_df

    if "Concurso" not in historical_df.columns:
        raise KeyError("Coluna 'Concurso' ausente no histórico; não é possível aplicar corte temporal.")

    filtered_df = historical_df[historical_df["Concurso"] < history_max_concurso]
    if filtered_df.empty:
        raise ValueError(
            "Nenhum histórico disponível antes do concurso alvo — verifique o CSV de concursos anteriores."
        )
    return filtered_df


def predict(
    input_file,
    model_file,
    output_file,
    duplo_strategy: str = "entropy",
    historical_file: str = "data/raw/concursos_anteriores.csv",
    historical_df: Optional[pd.DataFrame] = None,
    history_max_concurso: Optional[int] = None,
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
    duplo_strategy : str
        Strategy used to select the 5 games that will receive "duplos".
        Supported options:

        - ``"entropy"``: picks the five games with the highest entropy (default).
        - ``"top_margin"``: picks the five games with the lowest difference between the
          two most probable outcomes (``model_top_margin``), i.e., where the model is
          least confident.
        - ``"market"``: ignores the model confidence and always picks the single most
          likely outcome implied by the betting odds (baseline seco).
    historical_file : str
        Path of the CSV that contains past contests to build historical features.
        Ignored when ``historical_df`` is provided.
    historical_df : pandas.DataFrame, optional
        Pre-loaded historical dataset to avoid re-reading from disk.
    history_max_concurso : int, optional
        Exclusive upper bound applied to the "Concurso" column of the historical data
        to prevent temporal leakage during backtests.

    Returns
    -------
    None
        The predictions are saved to ``output_file``.
    """
    try:
        logging.info("Carregando dados dos jogos futuros...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        logging.info("Construindo histórico dos times a partir dos concursos anteriores...")
        if historical_df is None and not os.path.exists(historical_file):
            raise FileNotFoundError(
                f"Arquivo de histórico não encontrado em '{historical_file}'"
            )

        filtered_history_df = _load_historical_dataframe(
            historical_file,
            historical_df=historical_df,
            history_max_concurso=history_max_concurso,
        )
        history = build_team_history(filtered_history_df)

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

        # Gerar a coluna "Aposta"
        audit_columns = []
        missing_audit = []

        if duplo_strategy not in {"entropy", "top_margin", "market"}:
            raise ValueError(
                "Estrategia de duplos inválida. Use 'entropy', 'top_margin' ou 'market'."
            )

        if duplo_strategy == "market":
            logging.info(
                "Aplicando baseline de mercado: secos no maior P(·) implícito nas odds."
            )
            aposta_cols = ['P(1)', 'P(X)', 'P(2)']
            aposta_map = dict(zip(aposta_cols, ['1', 'X', '2']))
            df_features['Aposta'] = (
                df_features[aposta_cols]
                .idxmax(axis=1)
                .map(aposta_map)
            )
            df_features['Seco'] = df_features['Aposta']
        else:
            if duplo_strategy == "entropy":
                jogos_duplos_idxs = df_features.nlargest(5, 'Entropia').index
                logging.info(
                    "Selecionando duplos pelos maiores valores de entropia: %s",
                    jogos_duplos_idxs.tolist(),
                )
            else:
                jogos_duplos_idxs = df_features.nsmallest(5, 'model_top_margin').index
                logging.info(
                    "Selecionando duplos pelos menores valores de model_top_margin: %s",
                    jogos_duplos_idxs.tolist(),
                )

            # Gerar a coluna "Aposta" com 9 secos e 5 duplos
            logging.info("Gerando a coluna de aposta com 9 secos e 5 duplos...")
            df_features['Aposta'] = df_features['Seco']  # Copia as apostas secas inicialmente

            # Escolhendo os "duplos" para os 5 jogos mais incertos
            duplo_opcoes = ['1', 'X', '2']
            prob_matrix = ordered_probabilities.to_numpy()
            for idx in jogos_duplos_idxs:
                mais_provaveis = prob_matrix[idx].argsort()[-2:][::-1]  # Duas maiores probabilidades
                df_features.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"
        if missing_audit:
            raise KeyError(
                f"As seguintes colunas de auditoria estão ausentes no dataset: {missing_audit}"
            )

        ordered_cols = [
            'Concurso',
            'Jogo',
            'Mandante',
            'Visitante',
            'Aposta',
            'Seco',
            'Probabilidade (1)',
            'Probabilidade (X)',
            'Probabilidade (2)',
        ]

        audit_cols_order = [col for col in audit_columns if col in df_features.columns]
        remaining_cols = [
            col for col in df_features.columns if col not in ordered_cols + audit_cols_order
        ]

        final_columns = ordered_cols + audit_cols_order + remaining_cols

        # Salvando as predições no arquivo
        logging.info(f"Salvando predições no arquivo {output_file}...")
        df_features.to_csv(output_file, sep=';', index=False, columns=final_columns)
        logging.info(f"Previsões salvas com sucesso em {output_file}!")
    
    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
