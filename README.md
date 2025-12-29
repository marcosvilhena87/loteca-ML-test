# Loteca Machine Learning Pipeline

This project predicts results for the Brazilian "Loteca" football lottery. The repository contains a small data pipeline that cleans historical matches, trains a model and generates predictions for upcoming games.

## Requirements

- **Python**: 3.8 or newer

Install dependencies by running:

```bash
pip install -r requirements.txt
```

Run this command before executing the pipeline or running the tests.

## Data preparation

Place your CSV files in `data/raw/`:

- `concursos_anteriores.csv` – historical games with columns such as `Mandante`, `Visitante`, outcome flags (`[1]`, `[x]`, `[2]`) and odds.
- `proximo_concurso.csv` – upcoming games with the same odds columns.

Running the pipeline will create `data/processed/loteca_treinamento.csv` and store trained models in `models/`.

## Running the full pipeline

Execute the following from the repository root:

```bash
python main.py
```

The script processes the raw data, trains a `RandomForestClassifier` (for evaluation only) and writes predictions to
`output/predictions.csv`. The final bets follow a bookmaker-only, risk-controlled rule set:

- **Secos**: keep the favorite whenever its probability is at least 0.55.
- **Duplos**: pick five games only when the favorite and runner-up are close **and** the runner-up has at least 0.25
  probability. The primary filter is a small gap (≤ 0.12) plus `ProbSegundo ≥ 0.25`; if that yields fewer than five,
  only games with gap < 0.18 (still with `ProbSegundo ≥ 0.25`) are added. Any remaining slots stay as secos to avoid
  artificial pulverization. Each duplo uses the two highest bookmaker probabilities.
- **Antipulverização opcional**: if available, swap exactly one seco (that is not a duplo) for the second-highest
  outcome when the favorite is moderate (between 0.50 and 0.58) and the runner-up probability is at least 0.30.

## Running individual steps

Functions for each stage live inside `scripts/`. You can import and call them directly:

```bash
python - <<'PY'
from scripts import process, train, predict

process(
    'data/raw/concursos_anteriores.csv',
    'data/processed/loteca_treinamento.csv')
train(
    'data/processed/loteca_treinamento.csv',
    'models/final_model.pkl')
predict(
    'data/raw/proximo_concurso.csv',
    'models/final_model.pkl',
    'output/predictions.csv')
PY
```

This executes the same pipeline steps individually.

## Interpreting model performance

The classifier is trained **only on the bookmaker probabilities** (`P(1)`, `P(X)`, `P(2)`). Because the inputs contain no
extra signals, the best achievable accuracy is effectively the bookmaker argmax. In practice you should expect the
bookmaker baseline (always picking the highest probability) to sit near 0.52, with the RandomForest and a majority-class
baseline slightly below. This is not a bug—there is simply no additional information for a model to exploit.

For Loteca, raw accuracy is also a poor proxy for value. The pipeline is aimed at **expected value and controlled risk**:

- probabilities and gaps guide which games become duplos or secos;
- antipulverização optionally replaces one favorite when the contest looks highly concentrated;
- the goal is to surface cards with favorable expected payouts rather than to beat the bookmaker on accuracy alone.

## Running tests

The repository includes a few unit tests. Before running them, install all
project dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).
