
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
MODEL_FILE = os.path.join(UPLOAD_FOLDER, "agriculture_suitability_model.pkl")
ALLOWED_EXTENSIONS = {"pkl", "joblib"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS

def load_model():
    global model
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global model
    if 'model' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['model']
    if file and allowed_file(file.filename):
        fp = os.path.join(UPLOAD_FOLDER, "agriculture_suitability_model.pkl")
        file.save(fp)
        model = joblib.load(fp)
        return jsonify({"message": "Model uploaded and loaded"})
    return jsonify({"error": "Wrong file type"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        load_model()
        if model is None:
            return jsonify({"error": "No model loaded"}), 400
    try:
        data = request.get_json()
        features = {
            "Temperature": float(data.get("Temperature")),
            "Humidity": float(data.get("Humidity")),
            "Rainfall": float(data.get("Rainfall")),
            "SoilType": str(data.get("SoilType")),
            "CropType": str(data.get("CropType")),
            "FertilizerType": str(data.get("FertilizerType")),
            "PestInfestation": int(data.get("PestInfestation")),
        }
        import pandas as pd
        df = pd.DataFrame([features])
        pred = model.predict(df)[0]  # returns 0 or 1
        prob = float(getattr(model, "predict_proba", lambda x: [[None, 1.0]])(df)[0][1])
        return jsonify({
            "suitability": int(pred),
            "suitability_probability": round(prob, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5500, debug=True)