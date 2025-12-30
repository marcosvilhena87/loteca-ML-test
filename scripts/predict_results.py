import logging
import numpy as np
import pandas as pd
from joblib import load

from .features import (RatingEngine, compute_expert_differences,
                       compute_implied_probabilities, enrich_features)

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
            duo_alpha: float = DEFAULT_DUO_ALPHA):
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

        model = load(model_file)

        feature_columns = [
            "P_market(1)", "P_market(X)", "P_market(2)",
            "P_elo(1)", "P_elo(X)", "P_elo(2)",
            "P_pois(1)", "P_pois(X)", "P_pois(2)",
            "d_elo_1", "d_elo_X", "d_elo_2",
            "d_pois_1", "d_pois_X", "d_pois_2",
            "bookmaker_margin", "gap_market", "entropia_market", "draw_boost",
            "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
            "form_home", "form_away", "form_diff",
        ]

        X_future = future_df[feature_columns]
        probabilities = model.predict_proba(X_future)
        predictions = model.predict(X_future)

        classes = model.classes_
        prob_df = pd.DataFrame(probabilities, columns=[f"P_final({c})" for c in classes])
        future_df = pd.concat([future_df.reset_index(drop=True), prob_df], axis=1)
        future_df['Secos'] = predictions

        epsilon = 1e-10
        adjusted_probabilities = probabilities + epsilon
        future_df['entropia_final'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)
        top_two = np.sort(adjusted_probabilities, axis=1)[:, ::-1][:, :2]
        future_df['gap_final'] = top_two[:, 0] - top_two[:, 1]
        future_df['duplo_score'] = future_df['entropia_final'] + duo_alpha * (1 - future_df['gap_final'])

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
