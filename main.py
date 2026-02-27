#!/usr/bin/env python3
"""Pipeline principal: preprocessar, treinar e prever Loteca."""

from scripts.preprocess_data import main as preprocess_main
from scripts.train_model import main as train_main
from scripts.predict_results import main as predict_main


if __name__ == "__main__":
    preprocess_main()
    train_main()
    predict_main()
