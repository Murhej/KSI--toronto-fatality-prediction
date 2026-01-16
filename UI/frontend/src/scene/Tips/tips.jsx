import React, { useState } from "react";
import { Link } from "react-router-dom";
import logo from '../../Logo.png';
import '../PCRT.css';
import '../../App.css';

const safetyTips = [
  {
    title: "🚗 For Automobile Drivers",
    tips: [
      "Always wear your seatbelt and ensure passengers are buckled.",
      "Follow speed limits, especially in school zones and residential areas.",
      "Avoid distractions — no texting, eating, or GPS use while driving.",
      "Never drive under the influence of drugs or alcohol.",
      "Obey traffic signals and yield to pedestrians at crosswalks.",
      "Check mirrors and blind spots before changing lanes or reversing.",
      "Drive defensively — anticipate the actions of others."
    ]
  },
  {
    title: "🏍️ For Motorcyclists",
    tips: [
      "Wear a certified helmet and full protective gear.",
      "Use headlights and reflective clothing to stay visible.",
      "Ride within speed limits and road conditions.",
      "Stay out of vehicle blind spots and avoid weaving through traffic.",
      "Take safety training to boost your skill and confidence.",
      "Keep both hands on handlebars and feet on footrests.",
      "Be extra cautious in poor weather or on slippery roads."
    ]
  },
  {
    title: "🚴 For Cyclists",
    tips: [
      "Always wear a helmet and reflective gear.",
      "Use front and rear lights at night or in low visibility.",
      "Ride with traffic and use designated bike lanes when possible.",
      "Use hand signals to indicate turns and stops.",
      "Follow all traffic signs and rules — bikes are vehicles too.",
      "Avoid distractions like music or phones while riding.",
      "Be alert at intersections — don’t assume drivers see you."
    ]
  },
  {
    title: "🚶 For Pedestrians",
    tips: [
      "Cross at marked crosswalks or intersections only.",
      "Look both ways and make eye contact with drivers before crossing.",
      "Stay alert — don’t use your phone or wear headphones while walking.",
      "Wear visible or reflective clothing at night.",
      "Don’t assume vehicles will stop; wait for a full stop.",
      "Be careful near parking lots and driveways.",
      "Teach children safe pedestrian habits early on."
    ]
  },
  {
    title: "🧍‍♀️ For Passengers",
    tips: [
      "Always wear your seatbelt, even in the back seat.",
      "Don’t distract the driver — avoid loud or sudden actions.",
      "Help navigate only when it's safe to do so.",
      "Never rest feet on the dashboard — it’s a serious airbag risk.",
      "Check surroundings before opening doors near traffic.",
      "Verify your driver in rideshare apps before entering.",
      "Remain calm and respectful — it helps everyone stay safe."
    ]
  },
  {
    title: "🌙 Night Driving",
    tips: [
      "Ensure all lights are functioning and clean.",
      "Increase following distance and reduce speed.",
      "Be extra alert for pedestrians and animals.",
      "Avoid glare by dimming interior lights and using low beams appropriately."
    ]
  },
  {
    title: "🌅 Morning Driving",
    tips: [
      "Drive cautiously in wet or foggy early mornings.",
      "Beware of sun glare when heading east.",
      "Prepare your vehicle in advance for the day.",
      "Focus on the road and minimize distractions."
    ]
  },
  {
    title: "🕔 Rush Hour",
    tips: [
      "Maintain a safe distance and drive defensively.",
      "Stay calm in heavy, stop-and-go traffic.",
      "Use navigation apps to find alternate routes.",
      "Avoid distractions and keep your focus."
    ]
  },
  {
    title: "🎉 Weekend Driving",
    tips: [
      "Expect unpredictable traffic and plan for delays.",
      "Remain cautious even if traffic seems lighter.",
      "Check for local events or roadwork that may affect your route.",
      "Stay alert due to the potential for impaired drivers."
    ]
  },
  {
    title: "📅 Weekday Driving",
    tips: [
      "Anticipate busy roads during commutes.",
      "Be vigilant in school zones and business districts.",
      "Signal well before merging or exiting highways.",
      "Plan your route using real-time traffic updates."
    ]
  }
];

function Tips() {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleIndex = (index) => {
    setActiveIndex(index === activeIndex ? null : index);
  };

  return (
    <div>
      <header className="App-header">
        <img src={logo} alt='TFM LOGO' />
        <div className="button-container">
          <Link to="/">Home</Link>
          <Link to="/about">About Us</Link>
          <Link to="/tips">Safety Tips</Link>
        </div>
      </header>

      <main className="Tips-body">
        <div className="Tips-container">
          <h3 className="Tips-title">Safety Tips</h3>
          <div className="Tips-grid">
            {safetyTips.map((tipGroup, index) => (
              <div
                key={index}
                className={`Tips-card ${activeIndex === index ? "open" : ""}`}
              >
                <div className="Tips-header" onClick={() => toggleIndex(index)}>
                  <h4>{tipGroup.title}</h4>
                  <span className="Tips-toggle">{activeIndex === index ? "−" : "+"}</span>
                </div>
                {activeIndex === index && (
                  <ul className="Tips-list">
                    {tipGroup.tips.map((tip, i) => (
                      <li key={i}>{tip}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

export default Tips;