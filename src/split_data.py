import argparse
import os
import logging 
import pandas as pd
from get_data import read_params
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
filehandler = logging.FileHandler("./logs/split_data.logs")
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

def split_and_save(config_path):
    try:
        config = read_params(config_path)
        train_path = config["split_data"]["train_path"]
        test_path = config["split_data"]["test_path"]
        random_state = config["base"]["random_state"]
        test_size = config["split_data"]["test_size"]
        raw_dataset = config["load_data"]["raw_dataset_csv"]

        df = pd.read_csv(raw_dataset)
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        logger.info("split_and_save function runs successfully")
    
    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    try:
        args = argparse.ArgumentParser()
        args.add_argument("--config", default="params.yaml")
        parsed = args.parse_args()
        split_and_save(config_path=parsed.config)
        logger.info("split_data.py file runs successfully")
    except Exception as e:
        logger.error(e)