import logging

import numpy as np
import pandas as pd
from joblib import load

from .features import (RatingEngine, compute_expert_differences,
                       compute_implied_probabilities, enrich_features)
from .train_model import (CLASS_ORDER, DEFAULT_FEATURE_VARIANT, get_feature_columns,
                          _reorder_probas)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


DEFAULT_DUO_ALPHA = 0.5


def _load_history(history_file: str) -> pd.DataFrame:
    try:
        return pd.read_csv(history_file, delimiter=';', decimal='.')
    except FileNotFoundError:
        logging.warning(f"Arquivo de histórico {history_file} não encontrado. Continuando sem histórico.")
        return pd.DataFrame()


def predict(input_file, model_file, scaler_file=None, output_file=None, history_file: str = "data/processed/loteca_treinamento.csv",
            duo_alpha: float = DEFAULT_DUO_ALPHA, feature_variant: str = DEFAULT_FEATURE_VARIANT):
    """Generate predictions and rich diagnostics for future games."""
    try:
        logging.info("Carregando dados dos jogos futuros...")
        future_df = pd.read_csv(input_file, delimiter=';', decimal='.')

        required_columns = ['Odds 1', 'Odds X', 'Odds 2', 'Mandante', 'Visitante']
        missing_cols = [c for c in required_columns if c not in future_df.columns]
        if missing_cols:
            raise ValueError(f"Colunas necessárias ausentes em {input_file}: {missing_cols}")

        future_df = compute_implied_probabilities(future_df)

        history_df = _load_history(history_file)
        engine = RatingEngine()
        if not history_df.empty and {'Resultado', 'Mandante', 'Visitante'}.issubset(history_df.columns):
            logging.info("Atualizando estados de Elo/Poisson com histórico...")
            enrich_features(history_df, engine, update_results=True)
        else:
            logging.info("Histórico indisponível ou incompleto. Usando valores iniciais de Elo/Poisson.")

        logging.info("Gerando features para jogos futuros...")
        future_df = enrich_features(future_df, engine, update_results=False)
        future_df = compute_expert_differences(future_df)

        model_artifact = load(model_file)
        if isinstance(model_artifact, dict) and "model" in model_artifact:
            trained_variant = model_artifact.get("feature_variant", feature_variant)
            feature_columns = model_artifact.get("feature_columns", get_feature_columns(trained_variant))
            model = model_artifact["model"]
            if feature_variant != trained_variant:
                logging.warning("Sobrescrevendo variant treinada (%s) pela solicitada (%s).", trained_variant, feature_variant)
        else:
            model = model_artifact
            feature_columns = get_feature_columns(feature_variant)

        X_future = future_df[feature_columns]
        probabilities_raw = model.predict_proba(X_future)
        predictions = model.predict(X_future)

        classes = model.classes_
        probabilities = _reorder_probas(probabilities_raw, classes, CLASS_ORDER)
        prob_df = pd.DataFrame(probabilities, columns=[f"P_final({c})" for c in CLASS_ORDER])
        future_df = pd.concat([future_df.reset_index(drop=True), prob_df], axis=1)
        future_df['Secos'] = predictions
        future_df['Probabilidade (1)'] = future_df.get('P_final(1)', prob_df.get('P_final(1)'))
        future_df['Probabilidade (X)'] = future_df.get('P_final(X)', prob_df.get('P_final(X)'))
        future_df['Probabilidade (2)'] = future_df.get('P_final(2)', prob_df.get('P_final(2)'))
        future_df['Seco'] = future_df['Secos']

        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon
        future_df['entropia_final'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)
        top_two = np.sort(adjusted_probabilities, axis=1)[:, ::-1][:, :2]
        future_df['gap_final'] = top_two[:, 0] - top_two[:, 1]
        future_df['duplo_score'] = future_df['entropia_final'] + duo_alpha * (1 - future_df['gap_final'])
        future_df['Entropia'] = future_df['entropia_final']

        jogos_duplos_idxs = future_df.nlargest(5, 'duplo_score').index
        logging.info(f"Índices dos jogos mais incertos para duplos: {jogos_duplos_idxs.tolist()}")

        duplo_opcoes = list(classes)
        for idx in jogos_duplos_idxs:
            mais_provaveis = adjusted_probabilities[idx].argsort()[-2:][::-1]
            future_df.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"
        future_df['Aposta'] = future_df['Aposta'].fillna(future_df['Secos'])

        log_cols = ['Entropia', 'entropia_final', 'gap_market', 'gap_final', 'draw_boost']
        present_cols = [c for c in log_cols if c in future_df.columns]
        logging.info("Top-5 duplos e razões:")
        logging.info(future_df.loc[jogos_duplos_idxs, ['Aposta'] + present_cols])

        if output_file:
            logging.info(f"Salvando predições no arquivo {output_file}...")
            future_df.to_csv(output_file, sep=';', index=False)
            logging.info(f"Previsões salvas com sucesso em {output_file}!")

        return future_df

    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo não encontrado - {e}")
    except ValueError as e:
        logging.error(f"Erro: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
