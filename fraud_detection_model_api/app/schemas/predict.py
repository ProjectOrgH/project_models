from typing import Any, List, Optional

from pydantic import BaseModel
from titanic_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    {
                        "step": [1],
                        "type": ["PAYMENT"],
                        "amount": [9839.64],
                        "nameOrig": ["C1231006815"],
                        "oldbalanceOrg": [9839.64],
                        "newbalanceOrig": [170136.0],
                        "nameDest": ["M1979787155"],
                        "oldbalanceDest": [0.0],
                        "newbalanceDest": [0.0],
                        "isFraud": [0],
                        "isFlaggedFraud": [0],
                    }
                ]
            }
        }
