# app.py
import os
import io
import base64
import logging
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import pandas as pd
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Secret key: read from environment, never hardcode in source.
# Locally, set it via: export SECRET_KEY="some-random-string"
# On Render/production, set it in the platform's environment variable settings.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-insecure-key-do-not-use-in-production")

# Debug mode: OFF by default, only on if explicitly enabled via env var.
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_FOLDER, "sparq_rf_model.joblib")
ENCODER_PATH = os.path.join(MODEL_FOLDER, "complexity_encoder.joblib")

CORE_FEATURES = ['Duration', 'Team_Size', 'Budget', 'Past_Delays', 'Tech_Complexity']

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_pred_filename(filename):
    """Strip any existing 'pred_' prefix(es) before adding a new one,
    so re-uploading an already-predicted file doesn't produce
    pred_pred_pred_... filenames."""
    while filename.startswith("pred_"):
        filename = filename[len("pred_"):]
    return f"pred_{filename}"


def preprocess(df):
    # clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    # fill numeric na
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # encode complexity if present.
    # Use is_numeric_dtype rather than `dtype == 'object'` -- on some pandas
    # versions a text column reports as dtype('str'), not dtype('object'),
    # which silently skipped encoding and crashed the model downstream.
    if 'Tech_Complexity' in df.columns and not pd.api.types.is_numeric_dtype(df['Tech_Complexity']):
        if os.path.exists(ENCODER_PATH):
            le = joblib.load(ENCODER_PATH)
            # Guard against unseen categories at inference time instead of
            # letting LabelEncoder.transform raise and crash the request.
            known = set(le.classes_)
            df['Tech_Complexity'] = df['Tech_Complexity'].astype(str).apply(
                lambda v: v if v in known else le.classes_[0]
            )
            df['Tech_Complexity'] = le.transform(df['Tech_Complexity'])
        else:
            # No encoder available yet -- this should only happen before the
            # model has ever been trained via train_save.py.
            raise RuntimeError(
                "No complexity encoder found. Run train_save.py first to train "
                "the model and generate models/complexity_encoder.joblib."
            )
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

    if not allowed_file(uploaded.filename):
        flash("Only CSV files are supported", "danger")
        return redirect(url_for('index'))

    # Sanitize the filename before it touches the filesystem -- the raw
    # filename from a request is untrusted and can contain path-traversal
    # sequences like ../../ if not cleaned first.
    safe_name = secure_filename(uploaded.filename)
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    uploaded.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.exception("Failed to read uploaded CSV")
        flash(f"Could not read the uploaded CSV: {e}", "danger")
        return redirect(url_for('index'))

    # Model must already exist -- training happens only via train_save.py,
    # never as a side effect of a web request. This avoids a production
    # model silently being overwritten by whatever a user happens to upload.
    model = load_model()
    if model is None:
        flash("No trained model available yet. Run train_save.py to train and save a model first.", "danger")
        return redirect(url_for('index'))

    try:
        df = preprocess(df)
    except Exception as e:
        logger.exception("Preprocessing failed")
        flash(f"Could not process the uploaded CSV: {e}", "danger")
        return redirect(url_for('index'))

    features = [c for c in CORE_FEATURES if c in df.columns]
    if len(features) < 3:
        flash("CSV must include at least 3 of: Duration, Team_Size, Budget, Past_Delays, Tech_Complexity", "warning")
        return redirect(url_for('index'))

    # ---- Predict ----
    try:
        X_all = df[features]
        preds = model.predict(X_all)
    except Exception as e:
        logger.exception("Prediction failed")
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for('index'))

    df['Predicted_Risk'] = preds

    # ---- Table HTML ----
    sample_html = df.head(200).to_html(classes="table table-dark table-sm", index=False)

    # ---- Counts ----
    counts = df['Predicted_Risk'].value_counts().to_dict()

    # ---- Feature Importance ----
    try:
        importances = list(zip(features, model.feature_importances_))
        importances = sorted(importances, key=lambda x: x[1], reverse=True)
    except Exception:
        logger.warning("Could not compute feature importances", exc_info=True)
        importances = [(f, 0) for f in features]

    # -------- Generate Correlation Heatmap (PNG) ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    heatmap_base64 = None
    corr_columns = ['Duration', 'Team_Size', 'Budget', 'Past_Delays', 'Bug_Reports', 'Tech_Complexity']
    corr_columns = [c for c in corr_columns if c in df.columns]

    if len(corr_columns) > 1:
        try:
            plt.figure(figsize=(8, 5))
            sns.heatmap(df[corr_columns].corr(), annot=True, cmap="YlGnBu")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception:
            logger.warning("Could not generate heatmap", exc_info=True)

    # ---- Save predicted CSV (fixes the pred_pred_ filename bug) ----
    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)
    csv_bytes = out.getvalue().encode('utf-8')

    download_name = clean_pred_filename(safe_name)
    with open(os.path.join(UPLOAD_FOLDER, download_name), 'wb') as fh:
        fh.write(csv_bytes)

    # ---- Render dashboard ----
    return render_template(
        "dashboard.html",
        table_html=sample_html,
        counts=counts,
        importances=importances,
        total=len(df),
        high=int((df['Predicted_Risk'] == 'High').sum()),
        low=int((df['Predicted_Risk'] == 'Low').sum()),
        medium=int((df['Predicted_Risk'] == 'Medium').sum()),
        download_name=download_name,
        features=features,
        heatmap_base64=heatmap_base64,
    )


@app.route('/download/<filename>')
def download(filename):
    # secure_filename again on the way out, defense in depth against
    # a crafted filename in the URL path.
    safe_name = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    if not os.path.exists(path):
        flash("File not found", "danger")
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True)


@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Expected a JSON body"}), 400

    model = load_model()
    if model is None:
        return jsonify({"error": "Model not available. Run train_save.py to train a model first."}), 400

    try:
        df = pd.DataFrame([data])
        df = preprocess(df)
    except Exception as e:
        return jsonify({"error": f"Could not process input: {e}"}), 400

    features = [c for c in CORE_FEATURES if c in df.columns]
    missing = [f for f in CORE_FEATURES if f not in df.columns]
    if len(features) < 3:
        return jsonify({
            "error": "At least 3 of the following fields are required",
            "required_any_3_of": CORE_FEATURES,
            "missing": missing,
        }), 400

    try:
        X = df[features]
        pred = model.predict(X)[0]
    except Exception as e:
        logger.exception("Manual prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    return jsonify({"prediction": pred, "features_used": features})


if __name__ == "__main__":
    app.run(debug=DEBUG_MODE)
