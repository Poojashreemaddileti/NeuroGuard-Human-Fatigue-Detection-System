
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).parent.parent
DATA=ROOT/"data"/"training_features.csv"
OUT=ROOT/"models"/"fatigue_model.joblib"

df=pd.read_csv(DATA)
cols=["delta","theta","alpha","beta","label"]
missing=[c for c in cols if c not in df.columns]
if missing: raise ValueError(f"Missing columns: {missing}")
X=df[["delta","theta","alpha","beta"]]
y=df["label"].astype(int)
model=Pipeline([("scale",StandardScaler()),
 ("rf",RandomForestClassifier(n_estimators=400,random_state=42,class_weight="balanced"))])
model.fit(X,y)
joblib.dump(model,OUT)
print("Saved",OUT)
