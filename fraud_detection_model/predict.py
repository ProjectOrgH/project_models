import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
#print("os.getcwd():", os.getcwd())
import pandas as pd
import numpy as np

from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import config
from fraud_detection_model.pipeline import fraud_detection_pipe
from fraud_detection_model.processing.data_manager import load_pipeline
from fraud_detection_model.processing.data_manager import pre_pipeline_preparation
from fraud_detection_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
fraud_detection_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data: dict) -> dict:
    """Make a prediction using a saved model"""

    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    validated_data = validated_data.reindex(columns = config.model_config1.features)
    
    results = {
        "predictions": None,
        "version": _version,
        "errors": errors,
    }
    
    if not errors:
        predictions = fraud_detection_pipe.predict(validated_data)
        
        # Map boolean predictions to "Fraud" or "No Fraud"
        predictions_mapped = ["Fraud" if pred else "No Fraud" for pred in predictions]
        results = {"predictions": predictions_mapped, "version": _version, "errors": errors}
        
        #results = {"predictions": predictions, "version": _version, "errors": errors}
        print("results:",results)
    else:
        print("results:",results)
       
    return results

def make_prediction_bkp(*, input_data: dict) -> dict:
    """Make a prediction using a saved model"""
    
    data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    data = data.reindex(columns=["type", "amount", "oldbalanceOrg", "newbalanceOrig"])

    results = {
        "predictions": None,
        "version": _version,
    }

    predictions = fraud_detection_pipe.predict(data)

    results = {"predictions": predictions, "version": _version}
    print(results)

    return results


if __name__ == "__main__":

    data_in = {
        "step": [1],
        "type": ["TRANSFER"],
        "amount": [181.0],
        "nameOrig": ["C1305486145"],
        "oldbalanceOrg": [181.0],
        "newbalanceOrig": [0.0],
        "nameDest": ["C553264065"],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [0.0],
        "isFraud": [1],
        "isFlaggedFraud": [0],
    }

    make_prediction(input_data=data_in)
