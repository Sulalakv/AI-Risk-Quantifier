# train_save.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)

def preprocess(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    if 'Tech_Complexity' in df.columns and df['Tech_Complexity'].dtype == 'object':
        le = LabelEncoder()
        df['Tech_Complexity'] = le.fit_transform(df['Tech_Complexity'].astype(str))
        joblib.dump(le, 'models/complexity_encoder.joblib')
    return df

if __name__ == "__main__":
    df = pd.read_csv("SPARQ_training_data.csv")  # replace with your real labeled file
    df = preprocess(df)
    features = [c for c in ['Duration','Team_Size','Budget','Past_Delays','Tech_Complexity'] if c in df.columns]
    X = df[features]
    y = df['Risk_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/sparq_rf_model.joblib")
    print("Saved model -> models/sparq_rf_model.joblib")
