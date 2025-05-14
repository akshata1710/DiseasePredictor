import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import csv

st.set_page_config(page_title="Disease Predictor", layout="wide")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Advanced_MLP_Model.h5")
    return model

@st.cache_data
def load_data():
    desc_df = pd.read_csv("symptom_Description.csv")
    prec_df = pd.read_csv("symptom_precaution.csv")
    with open("symptoms_list.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        all_symptoms = [row[0] for row in reader]
    return desc_df, prec_df, all_symptoms

model = load_model()
desc_df, prec_df, all_symptoms = load_data()

diseases = { 0: "Tuberculosis", 1: "Cold", 2: "Influenza", 3: "Drug Reaction",
    4: "Malaria", 5: "Allergy", 6: "Hypothyroidism", 7: "Psoriasis",
    8: "GERD", 9: "Chronic cholestasis", 10: "Hepatitis A", 11: "Osteoarthritis",
    12: "(Vertigo) Paroymsal Positional Vertigo", 13: "Hypoglycemia",
    14: "Acne", 15: "Diabetes", 16: "Impetigo", 17: "Hypertension",
    18: "Peptic ulcer disease", 19: "Dimorphic hemorrhoids (piles)", 20: "Common Cold",
    21: "Chicken pox", 22: "Cervical spondylosis", 23: "Hyperthyroidism",
    24: "Urinary tract infection", 25: "Varicose veins", 26: "AIDS",
    27: "Paralysis (brain hemorrhage)" }

def preprocess(symptoms):
    input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    return np.array(input_vector).reshape(1, -1)

def predict_disease(selected_symptoms):
    input_data = preprocess(selected_symptoms)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    probability = float(prediction[0][predicted_class])

    disease_name = diseases.get(predicted_class, "Unknown Disease")

    desc_row = desc_df[desc_df["Disease"] == disease_name]
    description = desc_row["Description"].values[0] if not desc_row.empty else "Description not found."

    prec_row = prec_df[prec_df["Disease"] == disease_name]
    precautions = []
    if not prec_row.empty:
        for i in range(1, 5):
            p = prec_row.iloc[0].get(f"Precaution_{i}")
            if pd.notna(p):
                precautions.append(p)

    return disease_name, probability, description, precautions

st.title("🩺 Disease Prediction App (Direct TensorFlow Model)")
st.markdown("Select symptoms and click **Predict** to get disease diagnosis, precautions, and description.")

selected_symptoms = st.multiselect("Select your symptoms", options=all_symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        with st.spinner("Predicting disease..."):
            disease_name, probability, description, precautions = predict_disease(selected_symptoms)

        st.success(f"🦠 Predicted Disease: **{disease_name}**")
        st.markdown(f"🧪 Confidence: **{probability * 100:.2f}%**")

        st.markdown("### 📋 Description")
        st.info(description)

        if precautions:
            st.markdown("### 🛡️ Home Care Precautions")
            for tip in precautions:
                st.write(f"• {tip}")
