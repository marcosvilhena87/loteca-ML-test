import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .loteca_metrics import _ev_from_probabilities, evaluate_card
from .rateio_utils import load_rateio, sample_rateio_distribution

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


FEATURE_VARIANTS: Dict[str, List[str]] = {
    "market_plus_deltas": [
        "P_market(1)", "P_market(X)", "P_market(2)",
        "d_elo_1", "d_elo_X", "d_elo_2",
        "d_pois_1", "d_pois_X", "d_pois_2",
        "bookmaker_margin", "gap_market", "entropia_market", "draw_boost",
        "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
        "form_home", "form_away", "form_diff",
    ],
    "pure_probabilities": [
        "P_market(1)", "P_market(X)", "P_market(2)",
        "P_elo(1)", "P_elo(X)", "P_elo(2)",
        "P_pois(1)", "P_pois(X)", "P_pois(2)",
        "bookmaker_margin", "gap_market", "entropia_market",
        "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
        "form_home", "form_away", "form_diff",
    ],
    "full_mix": [
        "P_market(1)", "P_market(X)", "P_market(2)",
        "P_elo(1)", "P_elo(X)", "P_elo(2)",
        "P_pois(1)", "P_pois(X)", "P_pois(2)",
        "d_elo_1", "d_elo_X", "d_elo_2",
        "d_pois_1", "d_pois_X", "d_pois_2",
        "bookmaker_margin", "gap_market", "entropia_market", "draw_boost",
        "elo_diff", "elo_uncertainty_home", "elo_uncertainty_away",
        "form_home", "form_away", "form_diff",
    ],
}

DEFAULT_FEATURE_VARIANT = "market_plus_deltas"
DEFAULT_C_VALUES = (0.1, 0.3, 1.0, 3.0, 10.0)
LOGLOSS_TOL = 0.005
BRIER_TOL = 0.001
CLASS_ORDER = np.array(['1', 'X', '2'])
MIN_P14_MEAN = 5e-4
MIN_POSITIVE_EV_SHARE = 0.55
CUSTO_CARTAO = 3.0
DEFAULT_W14 = 1.0
DEFAULT_W13 = 1.0
MC_SAMPLES = 2000


def get_feature_columns(variant: str = DEFAULT_FEATURE_VARIANT) -> List[str]:
    if variant not in FEATURE_VARIANTS:
        raise KeyError(f"Feature variant desconhecida: {variant}. Opções: {list(FEATURE_VARIANTS)}")
    return FEATURE_VARIANTS[variant]


FEATURE_COLUMNS: List[str] = get_feature_columns()


def _time_ordered_split(df: pd.DataFrame, test_size: float = 0.2, embargo_contests: int = 1):
    df_sorted = df.sort_values(["Concurso", "Jogo"])
    split_index = int(len(df_sorted) * (1 - test_size))
    val_start_contest = df_sorted.iloc[split_index]["Concurso"]

    unique_contests = pd.Index(sorted(df_sorted["Concurso"].unique()))
    val_start_idx = unique_contests.get_loc(val_start_contest)
    embargo_cut = max(val_start_idx - embargo_contests, 0)

    train_contests = unique_contests[:embargo_cut]
    val_contests = unique_contests[val_start_idx:]

    train_df = df_sorted[df_sorted["Concurso"].isin(train_contests)]
    val_df = df_sorted[df_sorted["Concurso"].isin(val_contests)]

    logging.info(
        "Split temporal com embargo: %s concursos treino, %s concursos validação (embargo=%s)",
        len(train_contests), len(val_contests), embargo_contests,
    )
    return train_df, val_df


def _multiclass_brier(y_true: pd.Series, probas: np.ndarray, classes: np.ndarray) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[label] for label in y_true])
    y_onehot = np.eye(len(classes))[y_indices]
    return float(np.mean(np.sum((y_onehot - probas) ** 2, axis=1)))


def _reorder_probas(probas: np.ndarray, source_classes: Sequence[str], target_classes: Sequence[str]) -> np.ndarray:
    """Reorder probability columns to match the target class order."""

    index_map = [list(source_classes).index(cls) for cls in target_classes]
    return probas[:, index_map]


def _align_rateio(index: pd.Index, rateio_series: pd.Series | None) -> pd.Series:
    if rateio_series is None:
        return pd.Series(0.0, index=index)
    return rateio_series.reindex(index).fillna(0)


def train(
    input_file,
    model_file,
    scaler_file=None,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    c_values: Optional[Sequence[float]] = None,
    w14: float = DEFAULT_W14,
    w13: float = DEFAULT_W13,
):
    """Train a stacked meta-model on engineered features and persist artifacts."""
    try:
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        feature_columns = get_feature_columns(feature_variant)
        logging.info("Usando variant de features '%s' (%d colunas)", feature_variant, len(feature_columns))
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise KeyError(f"Features ausentes do conjunto processado: {missing_features}")

        logging.info("Separando treino e validação por ordem temporal com embargo...")
        train_df, val_df = _time_ordered_split(df, test_size=0.2, embargo_contests=1)
        X_train, X_val = train_df[feature_columns], val_df[feature_columns]
        y_train, y_val = train_df['Resultado'], val_df['Resultado']

        rateio_df = load_rateio("data/raw/concurso_rateio.csv", include_extended=True)
        rateio_unique = rateio_df.drop_duplicates(subset="Concurso")
        rateio_series = rateio_unique.set_index("Concurso")
        winsorized_rateio_14 = rateio_series.get("Rateio_14_winsor", rateio_series.get("Rateio_14"))
        winsorized_rateio_13 = rateio_series.get("Rateio_13_winsor", rateio_series.get("Rateio_13"))
        rateio_14_samples, rateio_13_samples = sample_rateio_distribution(rateio_df, n_samples=MC_SAMPLES)

        logging.info("Montando pipeline de mistura (logistic regression multiclasse)...")

        def _fit_and_score(C_value: float):
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=C_value, multi_class="multinomial", solver="lbfgs")),
            ])
            pipeline.fit(X_train, y_train)
            val_proba_raw = pipeline.predict_proba(X_val)
            labels = pipeline.classes_
            val_proba = _reorder_probas(val_proba_raw, labels, CLASS_ORDER)
            logloss_model = log_loss(y_val, val_proba, labels=CLASS_ORDER)
            brier_score = _multiclass_brier(y_val, val_proba, CLASS_ORDER)
            return {
                "model": pipeline,
                "val_proba": val_proba,
                "logloss_model": logloss_model,
                "brier": brier_score,
            }

        c_grid = list(dict.fromkeys(c_values if c_values is not None else DEFAULT_C_VALUES))
        if not c_grid:
            raise ValueError("A grade de valores de C não pode ser vazia.")
        market_cols = [f"P_market({c})" for c in CLASS_ORDER]
        val_market = val_df[market_cols].values
        logloss_market = log_loss(y_val, val_market, labels=CLASS_ORDER)
        brier_market = _multiclass_brier(y_val, val_market, CLASS_ORDER)

        val_eval_base = val_df.reset_index(drop=True)
        market_cols = [f"P_market({c})" for c in CLASS_ORDER]
        val_eval_base = val_eval_base.merge(
            rateio_df[["Concurso", "Rateio_14", "Rateio_13", "Acumulou_14"]],
            on="Concurso",
            how="left",
        )

        def _ev_medio(prob_series: pd.Series, rateio_series: pd.Series | None) -> float:
            aligned_rateio = _align_rateio(prob_series.index, rateio_series)
            return float((prob_series * aligned_rateio).mean())

        def _ev_total_por_concurso(
            p14_series: pd.Series,
            p13_series: pd.Series,
            rateio_14_series: pd.Series | None,
            rateio_13_series: pd.Series | None,
        ) -> pd.Series:
            rateio_14_aligned = _align_rateio(p14_series.index, rateio_14_series)
            rateio_13_aligned = _align_rateio(p13_series.index, rateio_13_series)
            return w14 * p14_series * rateio_14_aligned + w13 * p13_series * rateio_13_aligned - CUSTO_CARTAO

        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0]

        def _score_alphas(val_eval_df: pd.DataFrame, prob_cols: List[str]):
            alpha_records = []
            for alpha in alpha_grid:
                model_metrics = evaluate_card(val_eval_df, prob_cols, CLASS_ORDER, alpha=alpha)
                market_metrics = evaluate_card(val_eval_df, market_cols, CLASS_ORDER, alpha=alpha)
                ev14_model = _ev_medio(model_metrics.p14_by_contest, winsorized_rateio_14)
                ev14_market = _ev_medio(market_metrics.p14_by_contest, winsorized_rateio_14)
                ev13_model = _ev_medio(model_metrics.p13_by_contest, winsorized_rateio_13)
                ev13_market = _ev_medio(market_metrics.p13_by_contest, winsorized_rateio_13)
                ev_total_model = w14 * ev14_model + w13 * ev13_model - CUSTO_CARTAO
                ev_total_market = w14 * ev14_market + w13 * ev13_market - CUSTO_CARTAO
                ev_model_contest = _ev_total_por_concurso(
                    model_metrics.p14_by_contest, model_metrics.p13_by_contest, winsorized_rateio_14, winsorized_rateio_13
                )
                ev_market_contest = _ev_total_por_concurso(
                    market_metrics.p14_by_contest, market_metrics.p13_by_contest, winsorized_rateio_14, winsorized_rateio_13
                )
                ev_positive_share = float((ev_model_contest - ev_market_contest > 0).mean()) if not ev_model_contest.empty else 0.0
                ev_mc_mean, ev_mc_p10, ev_mc_p90 = _ev_from_probabilities(
                    model_metrics.p14_by_contest,
                    model_metrics.p13_by_contest,
                    rateio_14_samples,
                    rateio_13_samples,
                    w14,
                    w13,
                    CUSTO_CARTAO,
                )
                alpha_records.append({
                    "alpha": alpha,
                    "duplo_indices": dict(model_metrics.duplo_indices),
                    "ev14_modelo": ev14_model,
                    "ev14_mercado": ev14_market,
                    "ev13_modelo": ev13_model,
                    "ev13_mercado": ev13_market,
                    "ev_total_modelo": ev_total_model,
                    "ev_total_mercado": ev_total_market,
                    "ev_total_diff": ev_total_model - ev_total_market,
                    "ev_total_mc": ev_mc_mean,
                    "ev_total_p10": ev_mc_p10,
                    "ev_total_p90": ev_mc_p90,
                    "ev_positive_share": ev_positive_share,
                    "p14_medio_modelo": model_metrics.p14_medio,
                    "p14_medio_mercado": market_metrics.p14_medio,
                    "p13_medio_modelo": model_metrics.p13_medio,
                    "p13_medio_mercado": market_metrics.p13_medio,
                    "pct_13_modelo": model_metrics.survival.get(13, 0.0),
                    "pct_13_mercado": market_metrics.survival.get(13, 0.0),
                    "duplo_coverage_modelo": model_metrics.duplo_coverage,
                    "expected_hits_modelo": model_metrics.expected_hits,
                    "penalty_modelo": model_metrics.penalty,
                    "metrics_modelo": model_metrics,
                    "metrics_mercado": market_metrics,
                })

            alpha_df = pd.DataFrame(alpha_records)
            ordered = alpha_df.sort_values([
                "ev_total_diff", "ev_total_mc", "p14_medio_modelo", "ev_positive_share", "pct_13_modelo"
            ], ascending=False)
            best_row = ordered.iloc[0]
            return float(best_row["alpha"]), alpha_df, best_row["metrics_modelo"], best_row["metrics_mercado"]

        best_run = None
        all_runs = []
        for c_val in c_grid:
            run = _fit_and_score(c_val)
            delta = logloss_market - run["logloss_model"]

            val_prob_cols = [f"P_final({c})" for c in CLASS_ORDER]
            val_eval_df = pd.concat(
                [val_eval_base, pd.DataFrame(run["val_proba"], columns=val_prob_cols)],
                axis=1,
            )

            best_alpha, alpha_df, best_model_metrics, best_market_metrics = _score_alphas(val_eval_df, val_prob_cols)
            ev_model_14 = _ev_medio(best_model_metrics.p14_by_contest, winsorized_rateio_14)
            ev_market_14 = _ev_medio(best_market_metrics.p14_by_contest, winsorized_rateio_14)
            ev_model_13 = _ev_medio(best_model_metrics.p13_by_contest, winsorized_rateio_13)
            ev_market_13 = _ev_medio(best_market_metrics.p13_by_contest, winsorized_rateio_13)
            ev_model_total = w14 * ev_model_14 + w13 * ev_model_13 - CUSTO_CARTAO
            ev_market_total = w14 * ev_market_14 + w13 * ev_market_13 - CUSTO_CARTAO
            ev_model_contest = _ev_total_por_concurso(
                best_model_metrics.p14_by_contest, best_model_metrics.p13_by_contest, winsorized_rateio_14, winsorized_rateio_13
            )
            ev_market_contest = _ev_total_por_concurso(
                best_market_metrics.p14_by_contest, best_market_metrics.p13_by_contest, winsorized_rateio_14, winsorized_rateio_13
            )
            ev_positive_share = float((ev_model_contest - ev_market_contest > 0).mean()) if not ev_model_contest.empty else 0.0

            logging.info(
                "C=%.3g | LogLoss model=%.4f (delta=%.4f) | EV total diff=%.2f com alpha=%.2f",
                c_val, run["logloss_model"], delta, ev_model_total - ev_market_total, best_alpha,
            )
            logging.info("Duplos por alpha na validação: %s", best_model_metrics.duplo_indices)

            run_summary = {
                "C": c_val,
                **run,
                "best_alpha": best_alpha,
                "alpha_grid": alpha_df,
                "best_model_metrics": best_model_metrics,
                "best_market_metrics": best_market_metrics,
                "ev_total_diff": ev_model_total - ev_market_total,
                "ev_positive_share": ev_positive_share,
                "p14_model": best_model_metrics.p14_medio,
                "p14_market": best_market_metrics.p14_medio,
                "pct_13_model": best_model_metrics.survival.get(13, 0.0),
            }

            all_runs.append(run_summary)

        assert all_runs

        candidates = [
            r for r in all_runs
            if r["logloss_model"] <= logloss_market + LOGLOSS_TOL and r["brier"] <= brier_market + BRIER_TOL
        ]
        candidates = [
            r for r in candidates
            if r.get("p14_model", 0.0) >= MIN_P14_MEAN and r.get("ev_positive_share", 0.0) >= MIN_POSITIVE_EV_SHARE
        ]
        if not candidates:
            logging.warning(
                "Nenhum C respeitou os guarda-corpos (logloss e p14 mínimo/EV robusto). Usando melhor EV total disponível.",
                
            )
            candidates = all_runs

        def _run_sort_key(run):
            return (
                run["ev_total_diff"],
                run.get("p14_model", 0.0),
                run.get("ev_positive_share", 0.0),
                run.get("pct_13_model", 0.0),
            )

        best_run = sorted(candidates, key=_run_sort_key, reverse=True)[0]
        quality_best = min(all_runs, key=lambda r: (r["logloss_model"], r["brier"]))

        ev_model_total = _ev_total_por_concurso(
            best_run["best_model_metrics"].p14_by_contest,
            best_run["best_model_metrics"].p13_by_contest,
            winsorized_rateio_14,
            winsorized_rateio_13,
        )
        ev_market_total = _ev_total_por_concurso(
            best_run["best_market_metrics"].p14_by_contest,
            best_run["best_market_metrics"].p13_by_contest,
            winsorized_rateio_14,
            winsorized_rateio_13,
        )
        ev_model = float(ev_model_total.mean())
        ev_market = float(ev_market_total.mean())
        ev_model_mc, ev_model_p10, ev_model_p90 = _ev_from_probabilities(
            best_run["best_model_metrics"].p14_by_contest,
            best_run["best_model_metrics"].p13_by_contest,
            rateio_14_samples,
            rateio_13_samples,
            w14,
            w13,
            CUSTO_CARTAO,
        )
        ev_market_mc, ev_market_p10, ev_market_p90 = _ev_from_probabilities(
            best_run["best_market_metrics"].p14_by_contest,
            best_run["best_market_metrics"].p13_by_contest,
            rateio_14_samples,
            rateio_13_samples,
            w14,
            w13,
            CUSTO_CARTAO,
        )
        p14_model = best_run["p14_model"]
        p14_market = best_run["p14_market"]

        logging.info(
            "Melhor C=%.3g maximizando EV total (winsorizado) com guarda-corpo | EV diff=%.2f | LogLoss=%.4f (mercado=%.4f, melhor=%.4f)",
            best_run["C"], best_run["ev_total_diff"], best_run["logloss_model"], logloss_market, quality_best["logloss_model"],
        )
        logging.info(f"Brier Score no conjunto de validação: {best_run['brier']:.4f}")

        model_grid = best_run["alpha_grid"]

        logging.info(
            "Backtest de cartões (modelo vs mercado) maximizando EV total winsorizado:")
        for _, row in model_grid.iterrows():
            logging.info(
                "alpha=%.2f | EV diff=%.2f (modelo=%.0f vs mercado=%.0f) | EV_MC=%.0f p10=%.0f p90=%.0f | EV14=%.0f vs %.0f | EV13=%.0f vs %.0f | p14=%.3f vs mercado=%.3f | pct13 modelo=%.1f vs mercado=%.1f | share_EV>0=%.2f | cobertura=%.2f | EH=%.2f | Penalidade=%.2f",
                row["alpha"],
                row.get("ev_total_diff", float("nan")),
                row["ev14_modelo"],
                row["ev14_mercado"],
                row.get("ev_total_mc", float("nan")),
                row.get("ev_total_p10", float("nan")),
                row.get("ev_total_p90", float("nan")),
                row.get("ev14_modelo", float("nan")),
                row.get("ev14_mercado", float("nan")),
                row.get("ev13_modelo", float("nan")),
                row.get("ev13_mercado", float("nan")),
                row["p14_medio_modelo"],
                row["p14_medio_mercado"],
                row["pct_13_modelo"],
                row["pct_13_mercado"],
                row.get("ev_positive_share", float("nan")),
                row["duplo_coverage_modelo"],
                row["expected_hits_modelo"],
                row["penalty_modelo"],
            )

        logging.info(
            "Alpha ótimo=%.2f | Modelo: EV total=%.0f (MC=%.0f p10=%.0f p90=%.0f), p14=%.3f, pct13=%.1f, cobertura=%.2f, EH=%.2f, Penalidade=%.2f | Mercado EV total=%.0f (MC=%.0f p10=%.0f p90=%.0f), p14=%.3f, pct13=%.1f",
            best_run["best_alpha"],
            ev_model,
            ev_model_mc,
            ev_model_p10,
            ev_model_p90,
            p14_model,
            best_run["best_model_metrics"].survival.get(13, 0.0),
            best_run["best_model_metrics"].duplo_coverage,
            best_run["best_model_metrics"].expected_hits,
            best_run["best_model_metrics"].penalty,
            ev_market,
            ev_market_mc,
            ev_market_p10,
            ev_market_p90,
            p14_market,
            best_run["best_market_metrics"].survival.get(13, 0.0),
        )
        logging.info(f"Salvando o modelo em {model_file} (features={feature_variant})...")
        artifact = {
            "model": best_run["model"],
            "feature_variant": feature_variant,
            "feature_columns": feature_columns,
            "best_duplo_alpha": best_run["best_alpha"],
            "w14": w14,
            "w13": w13,
        }
        dump(artifact, model_file)

        if scaler_file:
            dump(None, scaler_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
        raise
    except KeyError as e:
        logging.error(f"Erro: {e}")
        raise
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        raise
