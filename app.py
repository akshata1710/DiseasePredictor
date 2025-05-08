from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np


# Load the trained model
model = None
try:
    with open("Advanced_MLP_Model.h5", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing to allow requests from your frontend

# Preprocessing function to handle symptoms input
def preprocess(symptoms):
    all_symptoms = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'fatigue',
        'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
        'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness',
        'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
        'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
        'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
        'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
        'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
        'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
        'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
        'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
        'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
        'bladder_discomfort', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
        'toxiclook(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
        'belly_pain', 'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
        'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
        'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
        'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
        'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
        'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]
    input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    return np.array(input_vector).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({"error": "Invalid input. Please provide a valid symptoms list."}), 400

        symptoms = data['symptoms']
        print(f"Received symptoms: {symptoms}")  # Debugging print to verify input data

        # Preprocess user input
        input_data = preprocess(symptoms)

        # Make prediction
        prediction = model.predict(input_data)

        # Extract the disease name and probability
        predicted_class = np.argmax(prediction)
        probability = prediction[0][predicted_class]

        # Example mapping for diseases (update to match model's output structure)
        diseases = {
            0: "Tuberculosis", 1: "Cold", 2: "Influenza", 3: "Drug Reaction", 4: "Malaria", 5: "Allergy",
            6: "Hypothyroidism", 7: "Psoriasis", 8: "GERD", 9: "Chronic cholestasis", 10: "Hepatitis A",
            11: "Osteoarthritis", 12: "(Vertigo) Paroymsal Positional Vertigo", 13: "Hypoglycemia", 14: "Acne",
            15: "Diabetes", 16: "Impetigo", 17: "Hypertension", 18: "Peptic ulcer disease", 19: "Dimorphic hemorrhoids (piles)",
            20: "Common Cold", 21: "Chicken pox", 22: "Cervical spondylosis", 23: "Hyperthyroidism",
            24: "Urinary tract infection", 25: "Varicose veins", 26: "AIDS", 27: "Paralysis (brain hemorrhage)"
        }
        disease_name = diseases.get(predicted_class, "Unknown Disease")

        # Mock description for each disease
        description = "This is a mock description for the predicted disease."

        # Return response
        return jsonify({
            'disease_name': disease_name,
            'probability': round(probability, 2),
            'description': description
        })

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Something went wrong on the server."}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
