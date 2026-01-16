import React, { useState } from "react";
import PCRT from '../Image/PoliceCRTLogo.png';
import PredictionResultDisplay from "../Components/PredictionResultDisplay";
import '../Components/PRD.css';
import TorontoMap from '../Components/TorontoMap';
import '../App.css';



function PolicCollisionRT({ location }) {
  const [month, setMonth] = useState('');
  const [dayOfWeek, setDayOfWeek] = useState('');
  const [timeOfDay, setTimeOfDay] = useState('');
  const [vehicleType, setVehicleType] = useState('');
  const [passenger, setPassenger] = useState(false);
  const [resultData, setResultData] = useState(null); // Holds the prediction result
  const [selecteAreaCode, setSelectedAreaCode] = useState(null);

  const handleSubmit = async () => {
    const requestData = {
      areaCode : selecteAreaCode,    // coming from props
      month,
      dayOfWeek,
      timeOfDay,
      vehicleType,
      passenger
    };

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData)
      });
      const result = await response.json();

      // Check for a success flag to decide if the result is valid.
      if (!result.success) {
        alert("Error: " + (result.error || "Unknown error"));
        console.log("Details:", result.details || result.errors);
        return;
      }
      // Instead of an alert, store the result so it can be displayed.
      setResultData(result);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert("Server error — check console.");
    }
  };

  const closePrediction = () => setResultData(null);

  return (
    <div className="criv-body">
      <div className="Logo-criv">
        <img src={PCRT} alt="Police CRT" />
        <h1>Police <br /><span>Collision Risk Tool</span></h1>
      </div>

      <div className="PCRT-Selection">
        <label htmlFor="Month">Select Time of Month:</label>
        <select id="Month" onChange={(e) => setMonth(e.target.value)}>
          <option value="">--Select--</option>
          {[
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
          ].map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <label htmlFor="DOW">Select Day of the Week:</label>
        <select id="DOW" onChange={(e) => setDayOfWeek(e.target.value)}>
          <option value="">--Select--</option>
          {[
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
          ].map(d => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>

        <label htmlFor="Day">Select Time of Day:</label>
        <select id="Day" onChange={(e) => setTimeOfDay(e.target.value)}>
          <option value="">--Select--</option>
          {["Morning", "Afternoon", "Evening", "Night"].map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <label htmlFor="Transportation">Type of Vehicle:</label>
        <select id="Transportation" onChange={(e) => setVehicleType(e.target.value)}>
          <option value="">--Select--</option>
          {["Automobile", "Motorcycle", "Bicycle", "Pedestrian"].map(v => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>

        <label>
          <input
            type="checkbox"
            checked={passenger}
            onChange={(e) => setPassenger(e.target.checked)}
          />
          Passenger
        </label>
      </div>
        <div className="Location-Body">
        
        
        <div className='Map-body'>
          <TorontoMap areaType="Police" onAreaSelect={setSelectedAreaCode} />
            </div>                            
        </div>
      <div className="criv-button">
        <button onClick={handleSubmit}>Predict Risk</button>
      </div>

      {resultData && (
        <div className="floating-result" onClick={closePrediction}>
          <div onClick={(e) => e.stopPropagation()}>
            <PredictionResultDisplay data={resultData} onClose={closePrediction} userType="police" />
          </div>
        </div>
      )}
    </div>
  );
}

export default PolicCollisionRT;
