from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from fraud_detection_model.config.core import config
from fraud_detection_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame = input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    step:Optional[int]
    type:Optional[str]
    amount:Optional[float]
    nameOrig:Optional[str]
    oldbalanceOrg:Optional[float]
    newbalanceOrig:Optional[float]
    nameDest:Optional[str]
    oldbalanceDest:Optional[float]
    newbalanceDest:Optional[float]
    isFraud:Optional[int]
    isFlaggedFraud:Optional[int] 


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]