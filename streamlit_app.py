import streamlit as st
import requests

# ✅ Must be at the very top
st.set_page_config(page_title="Disease Predictor", layout="wide")

# ✅ Load the full list of 222 symptoms from the CSV
with open("symptoms_list.csv", "r") as f:
    all_symptoms = [line.strip() for line in f if line.strip() and line.strip() != "symptom"]

st.title("🩺 Disease Prediction App (via Flask API)")
st.markdown("Select symptoms and click **Predict** to get disease diagnosis, precautions, and description.")
#st.write(f"✅ Total Symptoms Loaded: **{len(all_symptoms)}**")  # Should be 222

# User input: symptom selection
selected_symptoms = st.multiselect("Select your symptoms", options=all_symptoms)

# Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        try:
            with st.spinner("Sending symptoms to backend..."):
                # Send selected symptoms to Flask API
                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    json={"symptoms": selected_symptoms}
                )

            if response.status_code == 200:
                result = response.json()
                st.success(f"🦠 Predicted Disease: **{result['disease_name']}**")
                st.markdown(f"🧪 Confidence: **{result['probability']*100:.2f}%**")

                st.markdown("### 📋 Description")
                st.info(result.get("description", "No description available."))

                if result.get("home_care"):
                    st.markdown("### 🛡️ Home Care Precautions")
                    for tip in result["home_care"]:
                        st.write(f"• {tip}")

            else:
                st.error("❌ Failed to get a prediction from the API.")
        except Exception as e:
            st.error(f"⚠️ Could not connect to API:\n\n{e}")
