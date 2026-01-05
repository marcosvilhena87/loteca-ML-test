import logging

import numpy as np
import pandas as pd
from joblib import load

from .features import (RatingEngine, compute_expert_differences,
                       compute_implied_probabilities, enrich_features)
from .train_model import (
    CLASS_ORDER,
    CUSTO_CARTAO,
    DEFAULT_FEATURE_VARIANT,
    DEFAULT_W13,
    DEFAULT_W14,
    MC_SAMPLES,
    get_feature_columns,
    _reorder_probas,
)
from .loteca_metrics import _ev_from_probabilities, compute_hit_probabilities
from .rateio_utils import load_rateio, sample_rateio_distribution

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


DEFAULT_DUO_ALPHA = 0.5


def _load_history(history_file: str) -> pd.DataFrame:
    try:
        return pd.read_csv(history_file, delimiter=';', decimal='.')
    except FileNotFoundError:
        logging.warning(f"Arquivo de histórico {history_file} não encontrado. Continuando sem histórico.")
        return pd.DataFrame()


def predict(
    input_file,
    model_file,
    scaler_file=None,
    output_file=None,
    history_file: str = "data/processed/loteca_treinamento.csv",
    duo_alpha: float = DEFAULT_DUO_ALPHA,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    w14: float = DEFAULT_W14,
    w13: float = DEFAULT_W13,
    rateio_file: str = "data/raw/concurso_rateio.csv",
):
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
        rateio_df = load_rateio(rateio_file, include_extended=True)
        rateio_14_samples, rateio_13_samples = sample_rateio_distribution(rateio_df, n_samples=MC_SAMPLES)
        rateio_14_mean = rateio_df.get("Rateio_14_winsor", rateio_df.get("Rateio_14", pd.Series(dtype=float))).mean()
        rateio_13_mean = rateio_df.get("Rateio_13_winsor", rateio_df.get("Rateio_13", pd.Series(dtype=float))).mean()
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
            tuned_alpha = model_artifact.get("best_duplo_alpha")
            if tuned_alpha is not None and duo_alpha == DEFAULT_DUO_ALPHA:
                duo_alpha = tuned_alpha
            w14 = model_artifact.get("w14", w14)
            w13 = model_artifact.get("w13", w13)
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
        sorted_probs = np.sort(adjusted_probabilities, axis=1)[:, ::-1]
        p1 = sorted_probs[:, 0]
        p2 = sorted_probs[:, 1]

        future_df['gap_final'] = p1 - p2
        future_df['entropia_final'] = -np.sum(adjusted_probabilities * np.log(adjusted_probabilities), axis=1)
        future_df['Entropia'] = future_df['entropia_final']

        marginal_records = []
        for contest, contest_indices in future_df.groupby("Concurso").groups.items():
            contest_indices = list(contest_indices)
            base_probs = [float(p1[idx]) for idx in contest_indices]
            base_hits = compute_hit_probabilities(base_probs)
            base_ev_total = w14 * base_hits[14] * rateio_14_mean + w13 * base_hits[13] * rateio_13_mean - CUSTO_CARTAO

            for local_pos, idx in enumerate(contest_indices):
                duplo_prob = min(1.0, float(p1[idx] + p2[idx]))
                adjusted_probs = base_probs.copy()
                adjusted_probs[local_pos] = duplo_prob
                duplo_hits = compute_hit_probabilities(adjusted_probs)
                duplo_ev_total = w14 * duplo_hits[14] * rateio_14_mean + w13 * duplo_hits[13] * rateio_13_mean - CUSTO_CARTAO
                marginal_gain = duplo_ev_total - base_ev_total
                marginal_records.append((idx, marginal_gain, contest))

        marginal_df = pd.DataFrame(marginal_records, columns=["idx", "duplo_gain", "Concurso"])
        future_df['duplo_gain'] = marginal_df.set_index('idx')['duplo_gain']

        jogos_duplos_idxs = []
        for contest, contest_indices in future_df.groupby("Concurso").groups.items():
            contest_df = future_df.loc[contest_indices]
            contest_df = contest_df.sort_values(
                by=["duplo_gain", "entropia_final"], ascending=[False, True]
            )
            top_duplos = contest_df.head(5).index
            jogos_duplos_idxs.extend(top_duplos)

        logging.info(f"Índices dos jogos mais incertos para duplos: {sorted(jogos_duplos_idxs)}")

        duplo_opcoes = list(CLASS_ORDER)
        for idx in jogos_duplos_idxs:
            mais_provaveis = adjusted_probabilities[idx].argsort()[-2:][::-1]
            future_df.loc[idx, 'Aposta'] = f"{duplo_opcoes[mais_provaveis[0]]}, {duplo_opcoes[mais_provaveis[1]]}"
        future_df['Aposta'] = future_df['Aposta'].fillna(future_df['Secos'])

        log_cols = ['duplo_gain', 'Entropia', 'entropia_final', 'gap_market', 'gap_final', 'draw_boost']
        present_cols = [c for c in log_cols if c in future_df.columns]
        logging.info("Top-5 duplos e razões por concurso:")
        for contest, indices in future_df.groupby("Concurso").groups.items():
            contest_idxs = [idx for idx in jogos_duplos_idxs if idx in indices]
            logging.info(f"Concurso {contest}: {contest_idxs}")
            logging.info(future_df.loc[contest_idxs, ['Aposta'] + present_cols])

        contest_reports = []
        for contest, indices in future_df.groupby("Concurso").groups.items():
            base_probs = [float(p1[idx]) for idx in indices]
            hits = compute_hit_probabilities(base_probs)
            ev_samples = (
                w14 * np.array(rateio_14_samples) * hits[14]
                + w13 * np.array(rateio_13_samples) * hits[13]
                - CUSTO_CARTAO
            )
            duplos_contest = future_df.loc[future_df.index.isin(jogos_duplos_idxs)]
            duplos_contest = duplos_contest[duplos_contest["Concurso"] == contest]
            contest_reports.append(
                {
                    "Concurso": contest,
                    "EV_total_mean": float(ev_samples.mean()),
                    "EV_total_p10": float(np.percentile(ev_samples, 10)),
                    "EV_total_p90": float(np.percentile(ev_samples, 90)),
                    "EV14": float(hits[14] * rateio_14_mean * w14),
                    "EV13": float(hits[13] * rateio_13_mean * w13),
                    "P14": hits[14],
                    "P13": hits[13],
                    "Share_EV_pos": float((ev_samples > 0).mean()),
                    "Duplos": ", ".join(future_df.loc[idx, 'Aposta'] for idx in duplos_contest.index),
                    "Ganho_marginal_medio": float(future_df.loc[duplos_contest.index, 'duplo_gain'].mean()) if not duplos_contest.empty else 0.0,
                }
            )

        report_df = pd.DataFrame(contest_reports)
        logging.info("Resumo financeiro por concurso (EV total como objetivo):")
        logging.info(report_df)

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
