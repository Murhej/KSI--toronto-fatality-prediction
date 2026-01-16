import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import logo from './Logo.png';
import PolicCollisionRT from './scene/PoliceCollisionRT'
import CitizenRiskA from './scene/CitizenRiskA'

import './App.css';


function MainApp() {
  const [selectedRole, setSelectedRole] = useState(null);

  


  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} alt='TFM LOGO' />
        
        <div className="button-container">
          <Link to="/">Home</Link>
          <Link to="/about">About Us</Link>
          <Link to="/tips">Safety Tips</Link>
    
        </div>
      </header>


      <main className="App-body">
        <div className='Welcome-Body'>
          <h3>Welcome</h3>
          <h6 className='p-body'>
            Use this tool to assess fatal collision risk. Select your role, location,
            and conditions to predict risk and receive personalized safety tips.
          </h6>
        </div>
        <div className='Selection-Body'>
          <h3>Select Your Role</h3>
          <div className="role-options">
            <div className="role-card" onClick={() => setSelectedRole('Police')}>
              <i className="fa-solid fa-shield-halved"></i>
              <span>Police</span>
            </div>
            <div className="role-card" onClick={() => setSelectedRole('Citizen')}>
              <i className="fa-solid fa-user"></i>
              <span>Citizen</span>
            </div>
          </div>
        </div>
        
{selectedRole && (
  <div className={`predict-form ${selectedRole.toLowerCase()}-form`}>
    <div className="Form-body">
      <h3>
        <i className={`fa-solid ${selectedRole === 'Police' ? 'fa-shield-halved' : 'fa-user'}`}></i>
        {selectedRole}
      </h3>

      {selectedRole === 'Police' 
        ? <PolicCollisionRT  />
        : <CitizenRiskA />}
    </div>
  </div>
)}

        
      </main>

      <footer className='App-footer'>
        © 2025 Collision Risk AI. Empowering smarter, safer roads.
      </footer>
    </div>
  );
}

export default MainApp;
