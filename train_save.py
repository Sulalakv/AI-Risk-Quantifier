# train_save.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

os.makedirs("models", exist_ok=True)


def preprocess(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # Use is_numeric_dtype instead of `dtype == 'object'` -- on newer pandas
    # versions a text column can report as dtype('str') rather than
    # dtype('object'), which made the old check silently skip encoding and
    # crash model.fit() with "could not convert string to float".
    if 'Tech_Complexity' in df.columns and not pd.api.types.is_numeric_dtype(df['Tech_Complexity']):
        le = LabelEncoder()
        df['Tech_Complexity'] = le.fit_transform(df['Tech_Complexity'].astype(str))
        joblib.dump(le, 'models/complexity_encoder.joblib')
    return df


if __name__ == "__main__":
    df = pd.read_csv("SPARQ_training_data_5000.csv")  # replace with your real labeled file
    df = preprocess(df)

    features = [c for c in ['Duration', 'Team_Size', 'Budget', 'Past_Delays', 'Tech_Complexity'] if c in df.columns]
    X = df[features]
    y = df['Risk_Level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # ---- Evaluation on the held-out test set ----
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print(f"Labels order: {list(model.classes_)}")

    # ---- Cross-validation (more reliable than a single split) ----
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print("\n=== 5-Fold Cross-Validation ===")
    print(f"Scores: {np.round(cv_scores, 4)}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")

    # ---- Feature importance ----
    print("\n=== Feature Importance ===")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"{feat}: {imp:.4f}")

    # ---- Save the model trained on ALL data for production use ----
    # (train/test split above is only for honest evaluation; the shipped
    # model is refit on the full dataset so it doesn't waste 20% of the data)
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X, y)
    joblib.dump(final_model, "models/sparq_rf_model.joblib")

    # Persist evaluation metrics alongside the model so the README/app can reference them
    with open("models/evaluation_report.txt", "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Mean CV Accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))

    print("\nSaved model -> models/sparq_rf_model.joblib")
    print("Saved evaluation report -> models/evaluation_report.txt")