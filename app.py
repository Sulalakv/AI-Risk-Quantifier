# app.py
import os
import io
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "replace-this-if-deploying"
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_FOLDER, "sparq_rf_model.joblib")
ENCODER_PATH = os.path.join(MODEL_FOLDER, "complexity_encoder.joblib")

CORE_FEATURES = ['Duration','Team_Size','Budget','Past_Delays','Tech_Complexity']

def preprocess(df):
    # clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    # fill numeric na
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # encode complexity if present 
    if 'Tech_Complexity' in df.columns:
        if df['Tech_Complexity'].dtype == object:
            if os.path.exists(ENCODER_PATH):
                le = joblib.load(ENCODER_PATH)
                df['Tech_Complexity'] = le.transform(df['Tech_Complexity'].astype(str))
            else:
                le = LabelEncoder()
                df['Tech_Complexity'] = le.fit_transform(df['Tech_Complexity'].astype(str))
                joblib.dump(le, ENCODER_PATH)
    return df

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@app.route('/')
def index():
    model_exists = os.path.exists(MODEL_PATH)
    return render_template("index.html", model_exists=model_exists)


@app.route('/upload', methods=['POST'])
def upload():
    uploaded = request.files.get('file')
    if not uploaded or uploaded.filename == '':
        flash("No file uploaded", "danger")
        return redirect(url_for('index'))

    filepath = os.path.join(UPLOAD_FOLDER, uploaded.filename)
    uploaded.save(filepath)

    df = pd.read_csv(filepath)
    df = preprocess(df)

    # which features available
    features = [c for c in CORE_FEATURES if c in df.columns]
    if len(features) < 3:
        flash("CSV must include at least 3 of: Duration, Team_Size, Budget, Past_Delays, Tech_Complexity", "warning")
        return redirect(url_for('index'))

    # load model
    model = load_model()

    # train model if needed
    if (model is None) and ('Risk_Level' in df.columns):
        X = df[features]
        y = df['Risk_Level']
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)

    model = load_model()
    if model is None:
        flash("No pre-trained model available. Upload a labeled CSV (must include Risk_Level) to train.", "danger")
        return redirect(url_for('index'))

    # ---- Predict ----
    X_all = df[features]
    preds = model.predict(X_all)
    df['Predicted_Risk'] = preds

    # ---- Table HTML ----
    sample_html = df.head(200).to_html(classes="table table-dark table-sm", index=False)

    # ---- Counts ----
    counts = df['Predicted_Risk'].value_counts().to_dict()

    # ---- Feature Importance ----
    try:
        importances = list(zip(features, model.feature_importances_))
        importances = sorted(importances, key=lambda x: x[1], reverse=True)
    except:
        importances = [(f, 0) for f in features]

    # -------- Generate Correlation Heatmap (PNG) ----------
    import io, base64
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    heatmap_base64 = None

    corr_columns = ['Duration','Team_Size','Budget','Past_Delays','Bug_Reports','Tech_Complexity']
    corr_columns = [c for c in corr_columns if c in df.columns]

    if len(corr_columns) > 1:
        plt.figure(figsize=(8, 5))
        sns.heatmap(df[corr_columns].corr(), annot=True, cmap="YlGnBu")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # ---- Save predicted CSV ----
    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)
    csv_bytes = out.getvalue().encode('utf-8')

    with open(os.path.join(UPLOAD_FOLDER, f"pred_{uploaded.filename}"), 'wb') as fh:
        fh.write(csv_bytes)

    # ---- Render dashboard ----
    return render_template("dashboard.html",
                           table_html=sample_html,
                           counts=counts,
                           importances=importances,
                           total=len(df),
                           high=int((df['Predicted_Risk']=='High').sum()),
                           low=int((df['Predicted_Risk']=='Low').sum()),
                           medium=int((df['Predicted_Risk']=='Medium').sum()),
                           download_name=f"pred_{uploaded.filename}",
                           features=features,
                           heatmap_base64=heatmap_base64)

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        flash("File not found", "danger")
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.json
    # data expected keys matching CORE_FEATURES or at least available features
    df = pd.DataFrame([data])
    df = preprocess(df)
    features = [c for c in CORE_FEATURES if c in df.columns]
    model = load_model()
    if model is None:
        return jsonify({"error": "Model not available. Upload labeled CSV to train first."}), 400
    X = df[features]
    pred = model.predict(X)[0]
    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(debug=True)
