
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT=Path(__file__).parent.parent
MODEL_PATH=ROOT/"models"/"fatigue_model.joblib"

class FatigueModel:
    def __init__(self):
        self.model=None
        if MODEL_PATH.exists():
            try: self.model=joblib.load(MODEL_PATH)
            except Exception: self.model=None
        if self.model is None:
            self.model=self._demo()

    def _demo(self):
        rng=np.random.default_rng(42)
        X=[]; y=[]
        for label in range(3):
            for _ in range(240):
                b=rng.uniform(.75,1.25,4)
                if label==0: row=b*np.array([.90,.75,1.00,1.35])
                elif label==1: row=b*np.array([1.15,1.05,.82,.98])
                else: row=b*np.array([1.55,1.35,.56,.72])
                X.append(row/(row.sum() or 1)); y.append(label)
        pipe=Pipeline([("scale",StandardScaler()),
                       ("rf",RandomForestClassifier(n_estimators=220,random_state=42,class_weight="balanced"))])
        pipe.fit(X,y)
        return pipe

    def predict(self,features):
        p=int(self.model.predict([features])[0])
        prob=float(self.model.predict_proba([features])[0][p])
        return p,prob
