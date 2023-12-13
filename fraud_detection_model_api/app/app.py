from fastapi import FastAPI, Request, Response
import numpy as np

import gradio

from sklearn.metrics import f1_score, precision_score, recall_score
#import prometheus_client as prom
from fraud_detection_model.predict import make_prediction
from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import config
from fraud_detection_model.pipeline import fraud_detection_pipe
from fraud_detection_model.processing.data_manager import load_pipeline
from fraud_detection_model.processing.data_manager import pre_pipeline_preparation
from fraud_detection_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
fraud_detection_pipe = load_pipeline(file_name=pipeline_file_name)



app = FastAPI()

import pandas as pd

#test_data = pd.read_csv("test_reviews.csv")

#f1_metric = prom.Gauge('sentiment_f1_score', 'F1 score for random 100 test samples')
#precision_metric = prom.Gauge('sentiment_precision_score', 'Precision score for random 100 test samples')
#recall_metric = prom.Gauge('sentiment_recall_score', 'Recall score for random 100 test samples')


# Function for response generation
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

def predict_fraud(in_type='PAYMENT', in_amount=9839.64, in_oldbalanceOrig=9839.64, in_newbalanceOrig=170136.0):

    input = [type_map(in_type), in_amount, in_oldbalanceOrig, in_newbalanceOrig]

    input_to_model = np.array(input).reshape(1, -1)
    
    predictions = fraud_detection_pipe.predict(input_to_model)
    predictions_mapped = ["Fraud" if pred else "No Fraud" for pred in predictions]
    #results = {"predictions": predictions_mapped, "version": _version, "errors": errors}
    
    

    return predictions_mapped
    #print(result)

  #  if result[0]==1:
  #     return 'No'            # if DEATH_EVENT=1 means survive='No'
  #  else:
  #     return 'Yes'

# Function for updating metrics
#def update_metrics():
#    test = test_data.sample(100)
#    test_text = test['Text'].values
#    test_pred = sentiment_model(list(test_text))
#    pred_labels = [int(pred['label'].split("_")[1]) for pred in test_pred]
#    f1 = f1_score(test['labels'], pred_labels).round(3)
#    precision = precision_score(test['labels'], pred_labels).round(3)
#    recall = recall_score(test['labels'], pred_labels).round(3)
    
#    f1_metric.set(f1)
#    precision_metric.set(precision)
#    recall_metric.set(recall)


#@app.get("/metrics")
#async def get_metrics():
#    update_metrics()
#    return Response(media_type="text/plain", content= prom.generate_latest())
        
        
# Input from user
in_type = gradio.Radio(["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"], type="value",  label="Select type of transaction", show_label=True)
#in_amount = gradio.Number(default=0, label="Enter transaction amount", show_label=True)
#in_oldbalanceOrig = gradio.Textbox(type="float", default="0.0", label="What was the old balance at origin")
#in_newbalanceOrig = gradio.Textbox(type="float", default="0.0", label="What is the new balance at origin")

in_amount = gradio.Slider(minimum=0, maximum=100000, value=9839.64, step=1, label="Enter transaction amount", show_label=True)
in_oldbalanceOrig = gradio.Slider(minimum=0, maximum=100000, value=9839.64, step=1, label="What was the old balance at origin", show_label=True)
in_newbalanceOrig = gradio.Slider(minimum=0, maximum=100000, value=170136.0, step=1, label="What is the New balance at origin", show_label=True)


# Output response
out_response = gradio.components.Textbox(type="text", label='IsFraud')

# Gradio interface to generate UI link
title = "Fraud Detection"
description = "Fraud detection"

iface = gradio.Interface(fn = predict_fraud,
                         inputs = [in_type, in_amount, in_oldbalanceOrig, in_newbalanceOrig],
                         outputs = [out_response],
                         title = title,
                         description = description)

app = gradio.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)