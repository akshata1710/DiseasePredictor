from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pandas as pd
import csv

# ✅ Load the trained model
try:
    model = tf.keras.models.load_model("Advanced_MLP_Model.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ✅ Load disease description and precautions CSVs
try:
    desc_df = pd.read_csv("symptom_Description.csv")
    prec_df = pd.read_csv("symptom_precaution.csv")
    print("✅ Disease data loaded.")
except Exception as e:
    print(f"❌ Error loading disease data: {e}")
    desc_df, prec_df = pd.DataFrame(), pd.DataFrame()

# ✅ Load the correct list of 222 symptoms
with open("symptoms_list.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    all_symptoms = [row[0] for row in reader]


# ✅ Symptom preprocessing function
def preprocess(symptoms):
    input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    return np.array(input_vector).reshape(1, -1)


# ✅ Disease index mapping
diseases = {
    0: "Tuberculosis",
    1: "Cold",
    2: "Influenza",
    3: "Drug Reaction",
    4: "Malaria",
    5: "Allergy",
    6: "Hypothyroidism",
    7: "Psoriasis",
    8: "GERD",
    9: "Chronic cholestasis",
    10: "Hepatitis A",
    11: "Osteoarthritis",
    12: "(Vertigo) Paroymsal Positional Vertigo",
    13: "Hypoglycemia",
    14: "Acne",
    15: "Diabetes",
    16: "Impetigo",
    17: "Hypertension",
    18: "Peptic ulcer disease",
    19: "Dimorphic hemorrhoids (piles)",
    20: "Common Cold",
    21: "Chicken pox",
    22: "Cervical spondylosis",
    23: "Hyperthyroidism",
    24: "Urinary tract infection",
    25: "Varicose veins",
    26: "AIDS",
    27: "Paralysis (brain hemorrhage)",
}

# ✅ Flask app setup
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "No symptoms provided"}), 400

        symptoms = data["symptoms"]
        input_data = preprocess(symptoms)
        print("✅ Input shape:", input_data.shape)

        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        probability = float(prediction[0][predicted_class])

        disease_name = diseases.get(predicted_class, "Unknown Disease")

        # ✅ Lookup description
        desc_row = desc_df[desc_df["Disease"] == disease_name]
        description = (
            desc_row["Description"].values[0]
            if not desc_row.empty
            else "Description not found."
        )

        # ✅ Lookup precautions
        prec_row = prec_df[prec_df["Disease"] == disease_name]
        precautions = []
        if not prec_row.empty:
            for i in range(1, 5):
                p = prec_row.iloc[0].get(f"Precaution_{i}")
                if pd.notna(p):
                    precautions.append(p)

        return jsonify(
            {
                "disease_name": disease_name,
                "probability": round(probability, 2),
                "description": description,
                "home_care": precautions,
            }
        )

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
