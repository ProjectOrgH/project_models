import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import config
from fraud_detection_model.pipeline import fraud_detection_pipe
from fraud_detection_model.processing.data_manager import load_pipeline
from fraud_detection_model.processing.data_manager import pre_pipeline_preparation


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
fraud_detection_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: dict) -> dict:
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
        "type": ["PAYMENT"],
        "nameOrig": ["C1231006815"],
        "oldbalanceOrg": [9839.64],
        "newbalanceOrig": [170136.0],
        "nameDest": ["M1979787155"],
        "oldbalanceDest": [0.0],
        "newbalanceDest": [0.0],
        "isFraud": [0],
        "isFlaggedFraud": [0],
    }

    make_prediction(input_data=data_in)
