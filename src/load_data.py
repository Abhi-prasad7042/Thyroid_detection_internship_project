import os
from get_data import read_params, get_data
import argparse
import logging
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.impute import KNNImputer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler = logging.FileHandler("./logs/load_data.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# I you want know more about clean processing then go in notebooks folder and open Thyroid-train-and-evaluate file

def load_and_clean(config_path):
    try: 
        config = read_params(config_path)
        df = get_data(config_path)
        target = config["base"]["target_col"]

        # handling imbalanced dataset
        X = df.drop(target, axis=1)
        y = df[target]
        over_sampler = RandomOverSampler(random_state=42)
        X, y= over_sampler.fit_resample(X,y)
        df = X
        df[target] = y

        # droping unnecessary columns
        df.drop("TBG", axis=1, inplace=True)
        drop_col = []
        for feat in df.columns:
            if "y" in df[feat].values:
                drop_col.append(feat)

        df.drop(drop_col, axis=1, inplace=True)

        # handling categorical columns
        df["sex"] = df["sex"].map({"M":1, "F":0, "?":np.nan})

        missing_col = []
        for feat in df.columns:
            if "?" in df[feat].values:
                missing_col.append(feat)
        
        for feat in missing_col:
            df[feat] = df[feat].replace("?", np.nan).astype(float)

        for feat in df.columns:
            if "t" in df[feat].values:
                df[feat] = df[feat].map({"t":1, "f":0})
        
        df["class"] = df["class"].map({"hypothyroid":1, "negative":0})

        # handling missing values and using knnimputer for missing values

        df["age"] = df["age"].fillna(df["age"].median())

        imputer = KNNImputer()
        X = df.drop(target, axis=1)
        y = df[target]
        X = imputer.fit_transform(X)
        df = pd.DataFrame(X, columns=df.columns[:-1])
        df[target]= y

        logger.info("load_and_clean function runs successfully")
        return df

    except Exception as e:
        logger.error(e)


def savedata(config_path):
    try:
        config = read_params(config_path)
        raw_dataset_path = config["load_data"]["raw_dataset_csv"]
        df = load_and_clean(config_path)
        logger.info("savedata function runs successfully")
        df.to_csv(raw_dataset_path, index=False)
    except Exception as e:
        logger.error(e)

if __name__ =="__main__":
    try:
        args = argparse.ArgumentParser()
        args.add_argument("--config", default="params.yaml")
        parsed_args = args.parse_args()
        savedata(config_path=parsed_args.config)
        logger.info("load_data.py file runs successfully")
    except Exception as e:
        logger.error(e)