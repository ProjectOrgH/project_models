import sys
sys.path.append('/Users/ajaysingh/aimlops/Project/capstone_project/project_models')

import typing as t
from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

import os
from urllib.request import urlretrieve
import gdown

#print("Config:",config)
##  Pre-Pipeline Preparation

def type_map(transaction:str) -> int:  
    if transaction == 'CASH_OUT' :
        return 1
    elif transaction == 'PAYMENT' :
        return 2    
    elif transaction == 'CASH_IN' :
        return 3 
    elif transaction == 'TRANSFER' :
        return 4 
    elif transaction == 'DEBIT' :
        return 5 
    else:
        return 6
    
# 2. processing cabin

f1=lambda x: False if type(x) == float else True  
  

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame["type"] = data_frame["type"].apply(type_map)      

    data_frame['isFraud']=data_frame['isFraud'].apply(f1) 
                  
    #print("config.model_config:", config.model_config1)
    # drop unnecessary variables
    data_frame.drop(labels=config.model_config1.unused_fields, axis=1, inplace=True)

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    #print(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

def download_datafile(*, file_name: str, remote_url: str) -> None:
    # Download the file from the remote URL
    #urlretrieve(remote_url, f"{DATASET_DIR}/{file_name}")
    
    destination_path = DATASET_DIR / file_name
    
    # Ensure the parent directory exists
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the file using gdown
    with open(destination_path, 'wb') as f:
        gdown.download(remote_url, f, quiet=False)

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()