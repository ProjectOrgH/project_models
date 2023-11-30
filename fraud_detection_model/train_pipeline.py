import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from fraud_detection_model.config.core import config
from fraud_detection_model.pipeline import fraud_detection_pipe
from fraud_detection_model.processing.data_manager import load_dataset, save_pipeline, download_datafile

import os
from urllib.request import urlretrieve
from fraud_detection_model.config.core import DATASET_DIR

def run_training() -> None:
    
    """
    Train the model.
    """
    local_file_path = config.app_config.training_data_file
    file_path = DATASET_DIR / local_file_path
    
    
    if not os.path.isfile(file_path):
        #print(f"File not found at: {local_file_path}")
        #print("Downloading the file...")

        # Download the file from the remote URL
        #urlretrieve(remote_url, local_file_path)
        download_datafile(file_name=config.app_config.training_data_file,remote_url=config.app_config.remote_data_file_url)

        print("Download complete.")
    

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    print("Loading complete.")
    # divide train and test
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    
    fraud_detection_pipe.fit(x_train,y_train)  #
    fraud_detection_pipe.score(x_test,y_test)
    print("Accuracy(in %):", (fraud_detection_pipe.score(x_test,y_test))*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= fraud_detection_pipe)
    # printing the score
    
if __name__ == "__main__":
    run_training()