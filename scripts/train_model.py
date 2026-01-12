import logging
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from joblib import dump  # Usando joblib para salvar os modelos

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

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
            'p1_minus_p2',
            'confidence_gap',
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

        logging.info("Criando split temporal para calibração e tuning...")
        X_train_base, X_cal, y_train_base, y_cal, cal_contests = temporal_train_test_split(
            X_train, y_train, df.loc[X_train.index, "Concurso"], test_size=0.2
        )
        logging.info(
            "Calibração: treino até concurso %s, calibração a partir de %s.",
            cal_contests.max(),
            df.loc[X_cal.index, "Concurso"].min(),
        )

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

        # Tuning do C usando validação temporal
        logging.info("Avaliando valores de C para regularização...")
        candidate_cs = [0.05, 0.1, 0.3, 1, 3, 10]
        best_c = None
        best_logloss = np.inf
        for c_value in candidate_cs:
            candidate_model = LogisticRegression(
                random_state=42,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                C=c_value,
            )
            candidate_model.fit(X_tune_train, y_tune_train)
            val_probabilities = candidate_model.predict_proba(X_tune_val)
            val_logloss = log_loss(
                y_tune_val, val_probabilities, labels=candidate_model.classes_
            )
            logging.info("C=%s -> LogLoss validação: %.4f", c_value, val_logloss)
            if val_logloss < best_logloss:
                best_logloss = val_logloss
                best_c = c_value

        logging.info("Melhor C selecionado: %s", best_c)

        # Treinando o modelo base
        logging.info("Treinando o modelo base com C=%s...", best_c)
        model = LogisticRegression(
            random_state=42,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            C=best_c,
        )
        model.fit(X_train_base, y_train_base)

        # Calibração explícita com bloco temporal intermediário
        logging.info("Calibrando o modelo (isotonic)...")
        calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(model), method="isotonic"
        )
        calibrated_model.fit(X_cal, y_cal)

        # Avaliando o modelo
        probabilities = calibrated_model.predict_proba(X_test)
        predictions = calibrated_model.predict(X_test)
        classes = calibrated_model.classes_
        accuracy = accuracy_score(y_test, predictions)
        logloss = log_loss(y_test, probabilities, labels=classes)
        y_true = (
            pd.get_dummies(y_test)
            .reindex(columns=classes, fill_value=0)
            .to_numpy()
        )
        brier_score = np.mean(np.sum((probabilities - y_true) ** 2, axis=1)) / 2

        # Baseline usando probabilidades do mercado
        baseline_probabilities = X_test[['P(1)', 'P(X)', 'P(2)']].to_numpy()
        baseline_logloss = log_loss(y_test, baseline_probabilities, labels=classes)
        baseline_brier = (
            np.mean(np.sum((baseline_probabilities - y_true) ** 2, axis=1)) / 2
        )
        conf_matrix = confusion_matrix(y_test, predictions, labels=classes)
        logging.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")
        logging.info(f"Log loss no conjunto de teste: {logloss:.4f}")
        logging.info(f"Brier score no conjunto de teste: {brier_score:.4f}")
        logging.info(f"Baseline (mercado) Log loss: {baseline_logloss:.4f}")
        logging.info(f"Baseline (mercado) Brier score: {baseline_brier:.4f}")
        if baseline_logloss <= logloss and baseline_brier <= brier_score:
            logging.warning(
                "O baseline de mercado superou o modelo nas duas métricas "
                "(log loss e Brier)."
            )
        logging.info("Matriz de confusão no conjunto de teste:")
        logging.info("\n%s", conf_matrix)

        # Salvando o modelo
        logging.info(f"Salvando o modelo em {model_file}...")
        dump(calibrated_model, model_file)

        logging.info("Treinamento concluído com sucesso!")
    
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
