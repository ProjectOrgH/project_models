import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from fraud_detection_model.config.core import config

fraud_detection_pipe=Pipeline([
    
     ('model_dtc', DecisionTreeClassifier())
          
     ])