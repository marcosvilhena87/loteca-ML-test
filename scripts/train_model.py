import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .loteca_metrics import evaluate_card, summarize_alpha_grid
from .rateio_utils import load_rateio

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
CLASS_ORDER = np.array(['1', 'X', '2'])


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


def train(input_file, model_file, scaler_file=None, feature_variant: str = DEFAULT_FEATURE_VARIANT,
          c_values: Optional[Sequence[float]] = None):
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

        logging.info("Montando pipeline de mistura (logistic regression multiclasse)...")

        def _fit_and_score(C_value: float):
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=C_value)),
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

        best_run = None
        for c_val in c_grid:
            run = _fit_and_score(c_val)
            delta = logloss_market - run["logloss_model"]
            logging.info(
                "C=%.3g | LogLoss model=%.4f | delta vs mercado=%.4f | Brier=%.4f",
                c_val, run["logloss_model"], delta, run["brier"],
            )
            if best_run is None or run["logloss_model"] < best_run["logloss_model"]:
                best_run = {"C": c_val, **run}

        assert best_run is not None
        logging.info(
            "Melhor C=%.3g com LogLoss=%.4f (mercado=%.4f, delta=%.4f)",
            best_run["C"], best_run["logloss_model"], logloss_market, logloss_market - best_run["logloss_model"],
        )
        logging.info(f"Brier Score no conjunto de validação: {best_run['brier']:.4f}")

        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        val_prob_cols = [f"P_final({c})" for c in CLASS_ORDER]
        val_eval_df = pd.concat(
            [val_df.reset_index(drop=True), pd.DataFrame(best_run["val_proba"], columns=val_prob_cols)],
            axis=1,
        )

        def _proxy_ev_score(row: pd.Series) -> float:
            return (2 * row["pct_13"]) + row["pct_12"] + 0.3 * row["duplo_coverage"] + 0.1 * row["expected_hits"]

        def _select_best_alpha(grid_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
            scored = grid_df.assign(proxy_score=grid_df.apply(_proxy_ev_score, axis=1))
            if "ev14_medio" in scored and scored["ev14_medio"].notna().any():
                ordered = scored.sort_values(["ev14_medio", "pct_13", "pct_12"], ascending=False)
            else:
                ordered = scored.sort_values(["proxy_score", "pct_13", "pct_12"], ascending=False)
            return float(ordered.iloc[0]["alpha"]), scored

        rateio_df = load_rateio("data/raw/concurso_rateio.csv")
        val_eval_df = val_eval_df.merge(rateio_df[["Concurso", "Rateio_14", "Acumulou_14"]], on="Concurso", how="left")

        def _ev14_medio(metrics) -> float:
            rateio_series = (
                val_eval_df
                .drop_duplicates(subset="Concurso")
                .set_index("Concurso")["Rateio_14"]
            )
            aligned_rateio = rateio_series.reindex(metrics.p14_by_contest.index).fillna(0)
            return float((metrics.p14_by_contest * aligned_rateio).mean())

        model_grid = summarize_alpha_grid(
            val_eval_df, val_prob_cols, CLASS_ORDER, alpha_grid, rateio_df=val_eval_df
        )
        market_grid = summarize_alpha_grid(
            val_eval_df, [f"P_market({c})" for c in CLASS_ORDER], CLASS_ORDER, alpha_grid, rateio_df=val_eval_df
        )

        market_grid = market_grid.assign(proxy_score=market_grid.apply(_proxy_ev_score, axis=1))

        best_alpha, model_grid = _select_best_alpha(model_grid)
        best_metrics = evaluate_card(val_eval_df, val_prob_cols, CLASS_ORDER, alpha=best_alpha)
        market_metrics = evaluate_card(val_eval_df, [f"P_market({c})" for c in CLASS_ORDER], CLASS_ORDER, alpha=best_alpha)

        ev_model = _ev14_medio(best_metrics)
        ev_market = _ev14_medio(market_metrics)

        logging.info("Backtest de cartões (modelo vs mercado) usando proxy de EV (2*pct13 + pct12 + 0.3*cobertura + 0.1*EH):")
        for _, row in model_grid.merge(market_grid, on="alpha", suffixes=("_modelo", "_mercado")).iterrows():
            logging.info(
                "alpha=%.2f | proxy=%.2f | EV14=%.0f vs mercado=%.0f | %%>=13 modelo=%.1f (>=12=%.1f) vs mercado=%.1f (>=12=%.1f) | cobertura=%.2f | EH=%.2f",
                row["alpha"],
                row["proxy_score_modelo"],
                row.get("ev14_medio_modelo", 0),
                row.get("ev14_medio_mercado", 0),
                row["pct_13_modelo"],
                row["pct_12_modelo"],
                row["pct_13_mercado"],
                row["pct_12_mercado"],
                row["duplo_coverage_modelo"],
                row["expected_hits_modelo"],
            )

        logging.info(
            "Alpha ótimo=%.2f (proxy=%.2f) | Modelo: EV14=%.0f, %%>=13=%.1f, %%>=12=%.1f, cobertura=%.2f, EH=%.2f, Penalidade=%.2f | Mercado EV14=%.0f, %%>=13=%.1f",
            best_alpha,
            float(model_grid.loc[model_grid["alpha"] == best_alpha, "proxy_score"].iloc[0]),
            ev_model,
            best_metrics.survival.get(13, 0.0),
            best_metrics.survival.get(12, 0.0),
            best_metrics.duplo_coverage,
            best_metrics.expected_hits,
            best_metrics.penalty,
            ev_market,
            market_metrics.survival.get(13, 0.0),
        )

        logging.info(f"Salvando o modelo em {model_file} (features={feature_variant})...")
        artifact = {
            "model": best_run["model"],
            "feature_variant": feature_variant,
            "feature_columns": feature_columns,
            "best_duplo_alpha": best_alpha,
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
