import { Routes,Route } from 'react-router-dom';
import './App.css';
import Header from './components/Header';
import Homepage from './components/Homepage';
import Coinpage from './components/Coinpage';
import Notfound from './components/Notfound';
import ToggleColorMode from './components/ToggleColorMode';

function App() {
  return (
    <div className="App">
      <ToggleColorMode>
        <Header/>
        <Routes>
          <Route path="/" element={<Homepage />} exact />
          <Route path="/coins/:id" element={<Coinpage/>} />

          <Route path="*" element={<Notfound />} />

        </Routes>
      </ToggleColorMode>
      
    </div>
  );
}

export default App;
