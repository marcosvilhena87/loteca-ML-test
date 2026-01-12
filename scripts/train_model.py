import logging
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, recall_score
from joblib import dump  # Usando joblib para salvar os modelos

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def log_block_distribution(block_name, target_values):
    counts = target_values.value_counts()
    pct_2 = (target_values.astype(str) == "2").mean() * 100
    logging.info(
        "Distribuição de Resultado (%s): %s", block_name, counts.to_dict()
    )
    logging.info("Percentual de '2' em %s: %.2f%%", block_name, pct_2)


def log_prediction_diagnostics(stage_name, probabilities, predictions, classes):
    if "2" not in classes:
        logging.warning("Classe '2' não encontrada em %s.", stage_name)
        return
    class_index = list(classes).index("2")
    p2_values = probabilities[:, class_index]
    mean_p2 = float(np.mean(p2_values))
    median_p2 = float(np.median(p2_values))
    argmax_classes = classes[np.argmax(probabilities, axis=1)]
    argmax_pct = (argmax_classes == "2").mean() * 100
    prediction_counts = pd.Series(predictions).value_counts().to_dict()
    logging.info(
        "Diagnóstico %s: P(2) média=%.6f mediana=%.6f",
        stage_name,
        mean_p2,
        median_p2,
    )
    logging.info(
        "Diagnóstico %s: %% argmax=2 -> %.2f%%", stage_name, argmax_pct
    )
    logging.info(
        "Diagnóstico %s: contagem de previsões %s",
        stage_name,
        prediction_counts,
    )
    argmax_distribution = (
        pd.Series(argmax_classes)
        .value_counts(normalize=True)
        .mul(100)
        .reindex(classes, fill_value=0.0)
        .to_dict()
    )
    logging.info(
        "Diagnóstico %s: distribuição de argmax (%%) %s",
        stage_name,
        argmax_distribution,
    )


def log_class_recalls(stage_name, y_true, y_pred, classes):
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    recall_map = {label: float(score) for label, score in zip(classes, recalls)}
    logging.info("Recall por classe (%s): %s", stage_name, recall_map)


def log_low_prediction_share(stage_name, predictions, classes, threshold=0.05):
    prediction_share = (
        pd.Series(predictions)
        .value_counts(normalize=True)
        .reindex(classes, fill_value=0.0)
    )
    low_classes = prediction_share[prediction_share < threshold]
    if not low_classes.empty:
        pct_map = (low_classes * 100).round(2).to_dict()
        logging.warning(
            "Alerta %s: classes com < %.0f%% de predições: %s",
            stage_name,
            threshold * 100,
            pct_map,
        )


def evaluate_predictions(stage_name, y_true, probabilities, predictions, classes):
    accuracy = accuracy_score(y_true, predictions)
    logloss = log_loss(y_true, probabilities, labels=classes)
    y_true_bin = (
        pd.get_dummies(y_true)
        .reindex(columns=classes, fill_value=0)
        .to_numpy()
    )
    brier_score = np.mean(np.sum((probabilities - y_true_bin) ** 2, axis=1)) / 2
    conf_matrix = confusion_matrix(y_true, predictions, labels=classes)
    logging.info("Acurácia %s: %.4f", stage_name, accuracy)
    logging.info("Log loss %s: %.4f", stage_name, logloss)
    logging.info("Brier score %s: %.4f", stage_name, brier_score)
    logging.info("Matriz de confusão %s:", stage_name)
    logging.info("\n%s", conf_matrix)
    log_class_recalls(stage_name, y_true, predictions, classes)
    log_low_prediction_share(stage_name, predictions, classes)
    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier_score,
    }

def temporal_train_test_split(features, target, contest_series, test_size=0.2):
    contests = pd.Series(contest_series).dropna().astype(int)
    if contests.empty:
        raise ValueError("Não foi possível identificar o concurso para o split temporal.")
    unique_contests = np.sort(contests.unique())
    cutoff_index = max(int(len(unique_contests) * (1 - test_size)), 1)
    cutoff_contests = unique_contests[:cutoff_index]
    is_train = contests.isin(cutoff_contests)
    return (
        features.loc[is_train],
        features.loc[~is_train],
        target.loc[is_train],
        target.loc[~is_train],
        cutoff_contests,
    )


def train(input_file, model_file):
    """Train a classifier on processed data and persist artifacts.

    Parameters
    ----------
    input_file : str
        CSV file containing training features and labels.
    model_file : str
        Path to save the fitted model.
    Returns
    -------
    None
        The trained model is written to disk.
    """
    try:
        # Carregando os dados
        logging.info("Carregando os dados de entrada...")
        df = pd.read_csv(input_file, delimiter=';', decimal='.')

        # Selecionando as features (probabilidades) e o target (resultado real)
        logging.info("Selecionando as features e o target...")
        feature_columns = [
            'P(1)',
            'P(X)',
            'P(2)',
            'log_odds_1_2',
            'log_odds_1_X',
            'log_odds_X_2',
            'p1_minus_p2',
            'pX_minus_max',
            'fav_strength',
            'confidence_gap',
            'entropy_market',
        ]
        X = df[feature_columns]  # Features
        y = df['Resultado']  # Target: 1, X ou 2

        # Ordenar por concurso para o split temporal
        logging.info("Dividindo os dados em treino e teste (split temporal)...")
        df = df.sort_values(by=["Concurso", "Jogo"])
        X = df[feature_columns]
        y = df['Resultado']
        X_train, X_test, y_train, y_test, train_contests = temporal_train_test_split(
            X, y, df["Concurso"], test_size=0.2
        )
        logging.info(
            "Conjuntos temporais: treino até concurso %s, teste a partir de %s.",
            train_contests.max(),
            df.loc[X_test.index, "Concurso"].min(),
        )
        log_block_distribution("teste", y_test)

        logging.info("Criando split temporal para calibração e tuning...")
        X_train_base, X_cal, y_train_base, y_cal, cal_contests = temporal_train_test_split(
            X_train, y_train, df.loc[X_train.index, "Concurso"], test_size=0.2
        )
        logging.info(
            "Calibração: treino até concurso %s, calibração a partir de %s.",
            cal_contests.max(),
            df.loc[X_cal.index, "Concurso"].min(),
        )
        log_block_distribution("calibracao", y_cal)

        X_tune_train, X_tune_val, y_tune_train, y_tune_val, tune_contests = (
            temporal_train_test_split(
                X_train_base,
                y_train_base,
                df.loc[X_train_base.index, "Concurso"],
                test_size=0.2,
            )
        )
        logging.info(
            "Tuning: treino até concurso %s, validação a partir de %s.",
            tune_contests.max(),
            df.loc[X_tune_val.index, "Concurso"].min(),
        )
        log_block_distribution("tuning", y_tune_val)

        # Tuning do C e class_weight usando validação temporal
        logging.info("Avaliando valores de C para regularização...")
        candidate_cs = [0.05, 0.1, 0.3, 1, 3, 10]
        best_c = None
        best_class_weight = None
        best_logloss = np.inf
        candidate_class_weights = [
            None,
            {"1": 1.0, "X": 1.4, "2": 1.1},
        ]
        for class_weight in candidate_class_weights:
            for c_value in candidate_cs:
                candidate_model = LogisticRegression(
                    random_state=42,
                    solver="lbfgs",
                    max_iter=1000,
                    C=c_value,
                    class_weight=class_weight,
                )
                candidate_model.fit(X_tune_train, y_tune_train)
                val_probabilities = candidate_model.predict_proba(X_tune_val)
                val_logloss = log_loss(
                    y_tune_val, val_probabilities, labels=candidate_model.classes_
                )
                logging.info(
                    "class_weight=%s C=%s -> LogLoss validação: %.4f",
                    class_weight,
                    c_value,
                    val_logloss,
                )
                if val_logloss < best_logloss:
                    best_logloss = val_logloss
                    best_c = c_value
                    best_class_weight = class_weight

        logging.info(
            "Melhores parâmetros selecionados: C=%s class_weight=%s",
            best_c,
            best_class_weight,
        )

        # Treinando o modelo base
        logging.info(
            "Treinando o modelo base com C=%s e class_weight=%s...",
            best_c,
            best_class_weight,
        )
        model = LogisticRegression(
            random_state=42,
            solver="lbfgs",
            max_iter=1000,
            C=best_c,
            class_weight=best_class_weight,
        )
        model.fit(X_train_base, y_train_base)

        # Calibração explícita com bloco temporal intermediário
        logging.info("Calibrando o modelo (sigmoid)...")
        sigmoid_model = CalibratedClassifierCV(
            FrozenEstimator(model), method="sigmoid"
        )
        sigmoid_model.fit(X_cal, y_cal)

        isotonic_model = None
        min_cal_samples = 10
        cal_counts = y_cal.value_counts()
        if (cal_counts >= min_cal_samples).all():
            logging.info("Calibrando o modelo (isotonic)...")
            isotonic_model = CalibratedClassifierCV(
                FrozenEstimator(model), method="isotonic"
            )
            isotonic_model.fit(X_cal, y_cal)
        else:
            logging.warning(
                "Isotonic ignorado: calibração com poucas amostras por classe "
                "(mínimo=%s). Contagens: %s",
                min_cal_samples,
                cal_counts.to_dict(),
            )

        # Avaliando o modelo
        base_probabilities = model.predict_proba(X_test)
        base_predictions = model.predict(X_test)
        sigmoid_probabilities = sigmoid_model.predict_proba(X_test)
        sigmoid_predictions = sigmoid_model.predict(X_test)
        classes = model.classes_
        isotonic_probabilities = None
        isotonic_predictions = None
        if isotonic_model is not None:
            isotonic_probabilities = isotonic_model.predict_proba(X_test)
            isotonic_predictions = isotonic_model.predict(X_test)
        log_prediction_diagnostics(
            "base (sem calibração)",
            base_probabilities,
            base_predictions,
            classes,
        )
        log_prediction_diagnostics(
            "calibrado sigmoid",
            sigmoid_probabilities,
            sigmoid_predictions,
            classes,
        )
        if isotonic_model is not None:
            log_prediction_diagnostics(
                "calibrado isotonic",
                isotonic_probabilities,
                isotonic_predictions,
                classes,
            )

        evaluation_results = {}
        evaluation_results["base"] = evaluate_predictions(
            "base (sem calibração)",
            y_test,
            base_probabilities,
            base_predictions,
            classes,
        )
        evaluation_results["sigmoid"] = evaluate_predictions(
            "calibrado sigmoid",
            y_test,
            sigmoid_probabilities,
            sigmoid_predictions,
            classes,
        )
        if isotonic_model is not None:
            evaluation_results["isotonic"] = evaluate_predictions(
                "calibrado isotonic",
                y_test,
                isotonic_probabilities,
                isotonic_predictions,
                classes,
            )

        best_model_name = min(
            evaluation_results.items(), key=lambda item: item[1]["logloss"]
        )[0]
        if best_model_name == "base":
            final_model = model
        elif best_model_name == "isotonic":
            final_model = isotonic_model
        else:
            final_model = sigmoid_model
        logging.info("Modelo selecionado: %s", best_model_name)
        y_true = (
            pd.get_dummies(y_test)
            .reindex(columns=classes, fill_value=0)
            .to_numpy()
        )
        logloss = evaluation_results[best_model_name]["logloss"]
        brier_score = evaluation_results[best_model_name]["brier"]

        # Baseline usando probabilidades do mercado
        baseline_probabilities = X_test[['P(1)', 'P(X)', 'P(2)']].to_numpy()
        baseline_logloss = log_loss(y_test, baseline_probabilities, labels=classes)
        baseline_brier = (
            np.mean(np.sum((baseline_probabilities - y_true) ** 2, axis=1)) / 2
        )
        logging.info(f"Baseline (mercado) Log loss: {baseline_logloss:.4f}")
        logging.info(f"Baseline (mercado) Brier score: {baseline_brier:.4f}")
        if baseline_logloss <= logloss and baseline_brier <= brier_score:
            logging.warning(
                "O baseline de mercado superou o modelo nas duas métricas "
                "(log loss e Brier)."
            )

        # Salvando o modelo
        logging.info(f"Salvando o modelo em {model_file}...")
        dump(final_model, model_file)

        logging.info("Treinamento concluído com sucesso!")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
