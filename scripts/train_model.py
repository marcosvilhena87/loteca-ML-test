import logging
import pandas as pd
from joblib import dump  # Usando joblib para salvar os modelos
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

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

        # Selecionando features enriquecidas e o target (resultado real)
        logging.info("Selecionando as features e o target...")
        feature_cols = [
            'P(1)', 'P(X)', 'P(2)',
            'Pmax', 'Psecond', 'Gap', 'Entropy',
            'LogOdds_1', 'LogOdds_X', 'LogOdds_2',
            'DrawBias', 'DrawEntropyInteraction', 'DrawGapInteraction'
        ]
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Colunas de features ausentes no dataset processado: {missing}")

        X = df[feature_cols]
        y = df['Resultado']  # Target: 1, X ou 2

        # Dividindo os dados em treino e teste
        logging.info("Dividindo os dados em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Treinando o modelo com calibração de probabilidades
        logging.info("Treinando o modelo com calibração para probabilidades bem calibradas...")
        base_model = RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=None,
            class_weight="balanced_subsample",
        )
        model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
        model.fit(X_train, y_train)

        # Avaliando o modelo com foco em qualidade probabilística
        proba_pred = model.predict_proba(X_test)
        model_log_loss = log_loss(y_test, proba_pred, labels=model.classes_)

        y_test_one_hot = pd.get_dummies(y_test)
        y_test_one_hot = y_test_one_hot.reindex(columns=model.classes_, fill_value=0)
        brier_score = ((proba_pred - y_test_one_hot.values) ** 2).sum(axis=1).mean()

        logging.info(f"Log loss no conjunto de teste: {model_log_loss:.4f}")
        logging.info(f"Brier score no conjunto de teste: {brier_score:.4f}")

        # Baselines para comparação
        prob_cols = ['P(1)', 'P(X)', 'P(2)']
        bookmaker_prob = X_test[prob_cols].rename(columns={
            'P(1)': '1',
            'P(X)': 'X',
            'P(2)': '2'
        })
        bookmaker_prob = bookmaker_prob[model.classes_]
        bookmaker_log_loss = log_loss(y_test, bookmaker_prob, labels=model.classes_)
        bookmaker_brier = ((bookmaker_prob.values - y_test_one_hot.values) ** 2).sum(axis=1).mean()
        logging.info(f"Baseline bookmaker log loss: {bookmaker_log_loss:.4f}")
        logging.info(f"Baseline bookmaker Brier: {bookmaker_brier:.4f}")

        majority_class = y_train.value_counts().idxmax()
        majority_prob = pd.DataFrame(
            0,
            index=y_test.index,
            columns=model.classes_,
            dtype=float,
        )
        majority_prob[majority_class] = 1.0
        majority_log_loss = log_loss(y_test, majority_prob, labels=model.classes_)
        majority_brier = ((majority_prob.values - y_test_one_hot.values) ** 2).sum(axis=1).mean()
        logging.info(f"Baseline classe majoritária ({majority_class}) log loss: {majority_log_loss:.4f}")
        logging.info(f"Baseline classe majoritária ({majority_class}) Brier: {majority_brier:.4f}")

        # Salvando o modelo
        logging.info(f"Salvando o modelo em {model_file}...")
        dump(model, model_file)

        logging.info("Treinamento concluído com sucesso!")
    
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo {input_file} não foi encontrado.")
    except ValueError as e:
        logging.error(f"Erro nos dados: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
