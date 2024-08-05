import React, { useContext, useEffect } from 'react';
import {
  AppBar,
  Container,
  MenuItem,
  Select,
  Toolbar,
  Typography,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { CryptoState } from '../CryptoContext';
import { DarkModeSwitch } from 'react-toggle-dark-mode'; // Import DarkModeSwitch
import { styled } from '@mui/system';

import { ColorModeContext } from './ToggleColorMode'; // Import ColorModeContext

// Styled text component for animation
const AnimatedText = styled('div')`
  display: inline-block;
  cursor: pointer;
  font-size: 40px;
  text-transform: uppercase;
  color: goldenrod;
  span {
    display: inline-block;
    animation: flip 2s;
  }
  &:hover span {
    animation: none;
    animation: flip 2s;
  }
  @keyframes flip {
    0%, 80% {
      transform: rotateY(360deg);
    }
    100% {
      transform: rotateY(0deg);
    }
  }
`;

const Header = () => {
  const [isDarkMode, setDarkMode] = React.useState(true);
  
  const { currency, setCurrency } = CryptoState();
  const { toggleColorMode, mode } = useContext(ColorModeContext); // Use context for theme switching
  const navigate = useNavigate();

  const toggleDarkMode = (checked) => {
    toggleColorMode(checked);
    setDarkMode(checked);
  };

  // Update body class on mode change
  useEffect(() => {
    document.body.className = mode === 'dark' ? 'dark-mode' : 'light-mode';
  }, [mode]);

  return (
    <AppBar position="static" sx={{ backgroundColor: 'transparent' }}>
      <Container>
        <Toolbar sx={{ justifyContent: 'center' }}>
          <Typography
            onClick={() => navigate(`/`)}
            variant="h2"
            sx={{
              flexGrow: 1,
              color: 'goldenrod',
              fontFamily: 'Montserrat',
              fontWeight: 'bold',
              cursor: 'pointer',
              paddingLeft: '140px',
              textAlign: 'center',
            }}
          >
            <AnimatedText>
              {Array.from('Crypto Pulse').map((letter, index) => (
                <span key={index} style={{ '--i': index + 1 }}>
                  {letter === ' ' ? '\u00A0' : letter}
                </span>
              ))}
            </AnimatedText>
          </Typography>
          <Select
            variant="outlined"
            labelId="currency-select-label"
            id="currency-select"
            value={currency}
            onChange={(e) => setCurrency(e.target.value)}
            sx={{ width: 100, height: 40, marginLeft: 2 ,marginRight:4}}
          >
            <MenuItem value={'USD'}>USD</MenuItem>
            <MenuItem value={'INR'}>INR</MenuItem>
          </Select>
          <DarkModeSwitch
            checked={isDarkMode}
            onChange={toggleDarkMode}
            size={30} // Reduced size
          />
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;
