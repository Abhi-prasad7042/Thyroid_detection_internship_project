from random import random
import pandas as pd
import os
import json
from get_data import read_params
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler = logging.FileHandler("./logs/train_and_evaluate.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def scores_(actual, prediction):
    f1 = f1_score(actual, prediction)
    recall = recall_score(actual, prediction)
    precision = precision_score(actual, prediction)

    return f1, recall, precision


def train_and_evaluate(config_path):
    try:
        config = read_params(config_path)
        train_path = config['split_data']["train_path"]
        test_path = config["split_data"]["test_path"]
        test_size = config["split_data"]["test_size"]
        model_dir = config["model_dir"]
        scores_file = config["reports"]["scores"]
        params_file = config["reports"]["params"]

        criterion = config["estimators"]["RandomForestClassifier"]["params"]["criterion"]
        max_features = config["estimators"]["RandomForestClassifier"]["params"]["max_features"]
        min_samples_split = config["estimators"]["RandomForestClassifier"]["params"]["min_samples_split"]
        n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]

        target = config["base"]["target_col"]
        random_state = config["base"]["random_state"]

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        X_train = train.drop(target, axis=1)
        X_test = test.drop(target, axis=1)
        y_train = train[target]
        y_test = test[target]

        model = RandomForestClassifier(criterion=criterion, max_features=max_features, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        (f1, recall, precision) = scores_(y_test, y_pred)

        with open(scores_file, 'w') as f:
            scores = {
                "f1_score": f1,
                "Recall": recall,
                "Precision_score": precision
            }
            json.dump(scores, f, indent=4)

        with open(params_file, 'w') as f:
            params= {
                "n_estimators": n_estimators,
                "min_samples_split": min_samples_split,
                "max_features": max_features,  
                "criterion": criterion
            }
            json.dump(params, f, indent=4) 

        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        logger.info("train_And_evaluate function runs successfully")

    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    try:
        args = argparse.ArgumentParser()
        args.add_argument("--config", default="params.yaml")
        parsed_args = args.parse_args()
        train_and_evaluate(parsed_args.config)
        logger.info("train_and_evaluate.py file runs successfully")
    except Exception as e:
        logger.error(e)