import { Routes, Route } from 'react-router-dom'; 
import MainApp from './MainApp';
import AboutUs from './scene/aboutUs/AboutUs';
import Tips from './scene/Tips/tips';
import PoliceTips from './scene/Tips/PoliceTips';
import Map from './Components/TorontoMap'; // Capitalized

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainApp />} />
      <Route path="/about" element={<AboutUs />} />
      <Route path="/tips" element={<Tips />} />
      <Route path="/tipsPolice" element={<PoliceTips />} />
      <Route path="/map" element={<Map />} />
    
    </Routes>
  );
}

export default App;
