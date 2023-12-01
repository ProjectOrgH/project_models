import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

fraud_detection_pipe = Pipeline([("model_dtc", DecisionTreeClassifier())])
