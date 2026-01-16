import React, { useState } from "react";
import { Link } from "react-router-dom";
import logo from '../../Logo.png';
import '../PCRT.css';
import '../../App.css';

const officerTips = [
  {
    title: "⚠️ Dealing with Aggressive or Erratic Drivers",
    tips: [
      "Approach from a safe angle and maintain distance during stops.",
      "Use clear verbal commands and avoid escalating tone.",
      "Observe for signs of impairment, evasion, or mental instability.",
      "Position your cruiser for cover and backup if needed.",
      "Record the interaction via bodycam and dashcam where possible."
    ]
  },
  {
    title: "🏍️ Responding to High-Risk Motorcyclists",
    tips: [
      "Do not pursue at high speeds; collect plate/helmet description instead.",
      "Monitor intersections and known race zones for dangerous riders.",
      "Use aerial or roadside support when tracking reckless driving patterns.",
      "Anticipate sharp turns or evasive maneuvers."
    ]
  },
  {
    title: "🚶 Managing Dangerous Pedestrian Zones",
    tips: [
      "Increase patrol presence near late-night crosswalks and festivals.",
      "Watch for impaired or distracted walkers — especially with headphones or phones.",
      "Use audible sirens and spotlight sweeps in poorly lit areas.",
      "Educate pedestrians kindly but firmly about visibility and safe crossing."
    ]
  },
  {
    title: "🚨 When Responding to Fatality-Prone Zones",
    tips: [
      "Refer to real-time risk data and heatmaps when dispatching.",
      "Assess lighting, road width, and traffic conditions quickly on arrival.",
      "Use cones, reflective signage, and flares to secure the scene.",
      "Coordinate with traffic units and EMS for multi-vehicle incidents."
    ]
  },
  {
    title: "📍 Targeting High-Risk Time Periods",
    tips: [
      "Night: Expect speeding, impaired drivers, low visibility.",
      "Rush Hour: Prepare for aggressive lane changing, gridlock frustration.",
      "Weekend: Increased DUI likelihood — prioritize checkpoints and visibility.",
      "Morning: Fatigued or distracted commuters — monitor school zones heavily."
    ]
  },
  {
    title: "🎯 Enforcement and Prevention Tips",
    tips: [
      "Use unmarked vehicles in speed-risk areas for effective deterrence.",
      "Partner with community members to identify common crash sites.",
      "Set up data-driven patrol routes based on fatality probabilities.",
      "Educate drivers on-the-spot about high-risk behaviors during stops."
    ]
  },
  {
    title: "🧠 Officer Mental Health & De-escalation",
    tips: [
      "Take mental breaks between stressful responses when possible.",
      "Use calm, professional speech to defuse road rage incidents.",
      "Request assistance early — don’t try to handle volatile scenes solo.",
      "After traumatic crashes, report to peer support or wellness units."
    ]
  }
];

function PoliceTips() {
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
          <h3 className="Tips-title">🚓 Officer Safety & Response Tips</h3>
          <div className="Tips-grid">
            {officerTips.map((tipGroup, index) => (
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

export default PoliceTips;