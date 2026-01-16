import React, { useState } from "react";
import { Link } from "react-router-dom";
import logo from '../../Logo.png';
import '../PCRT.css';
import '../../App.css';

const aboutSections = [
  {
    title: "📘 About Collision Risk AI",
    content: [
      "Collision Risk AI is a predictive intelligence system designed to assess the risk of fatal traffic collisions in Toronto.",
      "Built to improve public safety, the platform empowers both law enforcement agencies and civilians with insight into when and where fatal incidents are most likely to occur.",
      "This system transforms raw historical traffic data into real-time insights."
    ]
  },
  {
    title: "🔍 How It Works",
    content: [
      "Evaluates factors such as location, day of the week, vehicle type, and time of day.",
      "Provides real-time risk predictions to help users make smarter travel or patrol decisions."
    ]
  },
  {
    title: "🎯 Purpose & Goals",
    content: [
      "Support proactive safety strategies for law enforcement by identifying high-risk zones and patterns.",
      "Enable the general public to make safer travel decisions with access to risk tools.",
      "Raise awareness of road safety risks through data-informed visuals and predictions.",
      "Promote smart city technology adoption to reduce preventable fatalities."
    ]
  },
  {
    title: "👥 Who It's For",
    content: [
      "Police & Public Safety: Identify danger zones and optimize patrol routes.",
      "Citizens: Estimate travel risk whether driving, walking, or biking.",
      "City Officials: Use insights for planning and public policy.",
      "Schools & Families: Encourage safe habits among students and young drivers."
    ]
  },
  {
    title: "📌 Why It Matters",
    content: [
      "Traffic collisions are a leading cause of injury and death in urban areas.",
      "By turning data into action, Collision Risk AI contributes to safer roads for everyone in Toronto.",
      "It’s designed to be user-friendly, practical, and informative."
    ]
  },
  {
    title: "🚀 Looking Ahead",
    content: [
      "Expand coverage across Ontario and Canada.",
      "Integrate real-time traffic and weather data.",
      "Introduce interactive heatmaps of high-risk areas.",
      "Support mobile platforms and multilingual access."
    ]
  },
  {
    title: "⚠️ Important Notice",
    content: [
      "This project is currently a demo.",
      "The predictions shown are for demonstration purposes only and should not be used for real-world decision-making.",
      "We are not responsible for any consequences resulting from reliance on this data. Always follow official road safety guidance."
    ]
  }
];

function AboutUs() {
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
          <h3 className="Tips-title">📖 About Us</h3>
          <div className="Tips-grid">
            {aboutSections.map((section, index) => (
              <div
                key={index}
                className={`Tips-card ${activeIndex === index ? "open" : ""}`}
              >
                <div className="Tips-header" onClick={() => toggleIndex(index)}>
                  <h4>{section.title}</h4>
                  <span className="Tips-toggle">{activeIndex === index ? "−" : "+"}</span>
                </div>
                {activeIndex === index && (
                  <ul className="Tips-list">
                    {section.content.map((item, i) => (
                      <li key={i}>{item}</li>
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

export default AboutUs;
