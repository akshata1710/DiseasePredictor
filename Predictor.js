import React, { useState } from 'react';
import axios from 'axios';

const Predictor = () => {
    const [symptoms, setSymptoms] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setResult(null);

        const symptomList = symptoms.split(',').map(s => s.trim().toLowerCase());
        try {
            const response = await axios.post('http://localhost:5000/predict', { symptoms: symptomList });
            setResult(response.data);
        } catch (err) {
            setError('Error fetching prediction. Please try again.');
        }
    };

    return (
        <div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'center' }}>
            <h2>Disease Prediction</h2>
            <form onSubmit={handleSubmit}>
                <input 
                    type="text" 
                    placeholder="Enter symptoms (comma separated)" 
                    value={symptoms} 
                    onChange={(e) => setSymptoms(e.target.value)}
                    style={{ width: '80%', padding: '8px', marginBottom: '10px' }}
                />
                <button type="submit" style={{ padding: '10px 15px', cursor: 'pointer' }}>Predict</button>
            </form>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            {result && (
                <div style={{ marginTop: '20px', textAlign: 'left', border: '1px solid #ccc', padding: '15px', borderRadius: '5px' }}>
                    <h3>Disease Name: {result.disease_name}</h3>
                    <p><strong>Probability:</strong> {result.probability}</p>
                    <p><strong>Description:</strong> {result.description}</p>
                    <p><strong>Recommended Home Care:</strong> {result.home_care}</p>
                </div>
            )}
        </div>
    );
};

export default Predictor;
