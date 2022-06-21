import os
import pandas as pd
import argparse
import logging
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler('./logs/get_data.logs')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def read_params(config_path):
    try: 
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logger.info("read params funtion runs successfully")
            return config
    except Exception as e:
        logger.error(e)

def get_data(config_path):
    try:
        config = read_params(config_path)
        datapath = config["data_source"]["s3_source"]
        df = pd.read_csv(datapath)
        logger.info("get_data funtion runs successfully")
        return df
    except Exception as e:
        logger.error(e)


if __name__=="__main__":
    try:
        parse = argparse.ArgumentParser()
        parse.add_argument("--config", default="params.yaml")
        parsed_args = parse.parse_args()
        logger.info("get_data.py  file runs successfully")
        data = get_data(config_path=parsed_args.config)
    except Exception as e:
        logger.error(e)