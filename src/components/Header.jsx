import React, { useContext, useEffect } from 'react';
import {
  AppBar,
  Container,
  MenuItem,
  Select,
  Toolbar,
  Typography,
  Button,
  Box,
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { CryptoState } from '../CryptoContext';
import { DarkModeSwitch } from 'react-toggle-dark-mode';
import { styled } from '@mui/system';
import { ColorModeContext } from './ToggleColorMode';

const AnimatedText = styled('div')`
  display: inline-block;
  cursor: pointer;
  font-size: 32px;
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
  const { toggleColorMode, mode } = useContext(ColorModeContext);
  const navigate = useNavigate();
  const location = useLocation();

  const toggleDarkMode = (checked) => {
    toggleColorMode(checked);
    setDarkMode(checked);
  };

  useEffect(() => {
    document.body.className = mode === 'dark' ? 'dark-mode' : 'light-mode';
  }, [mode]);

  const navItems = [
    { label: 'Markets', path: '/' },
    { label: 'LLM Signals', path: '/signals' },
    { label: 'Paper Portfolio', path: '/portfolio' },
    { label: 'Backtest Report', path: '/backtest' },
  ];

  return (
    <AppBar position="static" sx={{ backgroundColor: 'transparent', boxShadow: 'none' }}>
      <Container maxWidth="xl">
        <Toolbar sx={{ justifyContent: 'space-between', flexWrap: 'wrap', py: 1 }}>
          <Typography
            onClick={() => navigate(`/`)}
            variant="h4"
            sx={{
              color: 'goldenrod',
              fontFamily: 'Montserrat',
              fontWeight: 'bold',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
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

          <Box sx={{ display: 'flex', gap: 1, my: { xs: 1, md: 0 } }}>
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  sx={{
                    color: isActive ? 'goldenrod' : 'text.primary',
                    fontFamily: 'Montserrat',
                    fontWeight: isActive ? 700 : 500,
                    borderBottom: isActive ? '2px solid goldenrod' : '2px solid transparent',
                    borderRadius: 0,
                    px: 2,
                    '&:hover': {
                      color: 'goldenrod',
                      backgroundColor: 'rgba(218, 165, 32, 0.08)',
                    },
                  }}
                >
                  {item.label}
                </Button>
              );
            })}
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Select
              variant="outlined"
              labelId="currency-select-label"
              id="currency-select"
              value={currency}
              onChange={(e) => setCurrency(e.target.value)}
              sx={{ width: 90, height: 38, marginRight: 2 }}
            >
              <MenuItem value={'USD'}>USD</MenuItem>
              <MenuItem value={'INR'}>INR</MenuItem>
            </Select>
            <DarkModeSwitch
              checked={isDarkMode}
              onChange={toggleDarkMode}
              size={28}
            />
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;
