import React from 'react';
import { Link } from 'react-router-dom';
import './PRD.css';

function PredictionResultDisplay({ data, onClose, userType }) {
  if (!data || !data.success) {
    return (
      <div className="prediction-result-display error">
        <h2>Error: Prediction Failed</h2>
        <p>{data?.error || "An unknown error occurred."}</p>
        <button onClick={onClose} className="close-btn">Close</button>
      </div>
    );
  }

  const { probability, raw_prediction, classification } = data;

  return (
    <div className="floating-result">
      <div className="prediction-result-display styled-card">
        <h1>📊 Prediction Result</h1>

        <div className="prediction-values">
          <p><strong>Probability:</strong> {probability}</p>
          <p><strong>Classification:</strong> {classification}</p>
          <p><strong>Raw Model Label:</strong> {raw_prediction}</p>
        </div>

        <div className="width-safe">
          {userType === "police" ? (
            <Link to="/tipsPolice" className="tips-link">🚓 Officer Safety Tips</Link>
          ) : (
            <Link to="/tips" className="tips-link">🛡️ Safety Tips</Link>
          )}
        </div>

        <button onClick={onClose} className="close-btn styled-btn">Close</button>
      </div>
    </div>
  );
}

export default PredictionResultDisplay;
