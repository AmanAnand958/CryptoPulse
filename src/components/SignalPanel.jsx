import React, { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  CircularProgress,
  Chip,
  Tooltip,
  IconButton,
  LinearProgress,
} from "@mui/material";

// Inline SVG icons to avoid @mui/icons-material version conflicts
const RefreshIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M17.65 6.35A7.958 7.958 0 0 0 12 4C7.58 4 4 7.58 4 12s3.58 8 8 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0 1 12 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
  </svg>
);
const TrendingUpIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="m16 6 2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>
  </svg>
);
const TrendingDownIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="m16 18 2.29-2.29-4.88-4.88-4 4L2 7.41 3.41 6l6 6 4-4 6.3 6.29L22 12v6z"/>
  </svg>
);
const RemoveIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 13H5v-2h14v2z"/>
  </svg>
);

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
const REFRESH_INTERVAL_MS = 60_000; // 1 minute

const DIRECTION_CONFIG = {
  long: {
    label: "LONG",
    color: "#00e676",
    bg: "rgba(0,230,118,0.12)",
    Icon: TrendingUpIcon,
    emoji: "🟢",
  },
  short: {
    label: "SHORT",
    color: "#ff1744",
    bg: "rgba(255,23,68,0.12)",
    Icon: TrendingDownIcon,
    emoji: "🔴",
  },
  hold: {
    label: "HOLD",
    color: "#78909c",
    bg: "rgba(120,144,156,0.12)",
    Icon: RemoveIcon,
    emoji: "⚪",
  },
};

const COIN_LOGOS = {
  bitcoin: "https://assets.coingecko.com/coins/images/1/small/bitcoin.png",
  ethereum: "https://assets.coingecko.com/coins/images/279/small/ethereum.png",
  solana: "https://assets.coingecko.com/coins/images/4128/small/solana.png",
  binancecoin: "https://assets.coingecko.com/coins/images/825/small/bnb-icon2_2x.png",
  cardano: "https://assets.coingecko.com/coins/images/975/small/cardano.png",
};

function ConfidenceBar({ value, color }) {
  return (
    <Box sx={{ mt: 0.5 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.3 }}>
        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.7rem" }}>
          Confidence
        </Typography>
        <Typography variant="caption" sx={{ color, fontWeight: 700, fontSize: "0.7rem" }}>
          {Math.round(value * 100)}%
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={value * 100}
        sx={{
          height: 5,
          borderRadius: 3,
          backgroundColor: "rgba(255,255,255,0.08)",
          "& .MuiLinearProgress-bar": {
            backgroundColor: color,
            borderRadius: 3,
          },
        }}
      />
    </Box>
  );
}

function SignalCard({ signal }) {
  const cfg = DIRECTION_CONFIG[signal.direction] || DIRECTION_CONFIG.hold;
  const DirectionIcon = cfg.Icon;
  const logo = COIN_LOGOS[signal.coin];

  return (
    <Box
      id={`signal-card-${signal.coin}`}
      sx={{
        background: "linear-gradient(145deg, rgba(30,30,40,0.95), rgba(20,20,30,0.95))",
        border: `1px solid ${cfg.color}33`,
        borderRadius: 3,
        p: 2.5,
        transition: "all 0.3s ease",
        cursor: "default",
        "&:hover": {
          border: `1px solid ${cfg.color}88`,
          boxShadow: `0 4px 24px ${cfg.color}22`,
          transform: "translateY(-2px)",
        },
      }}
    >
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1.5 }}>
        {logo && (
          <img
            src={logo}
            alt={signal.symbol}
            width={32}
            height={32}
            style={{ borderRadius: "50%" }}
          />
        )}
        <Box>
          <Typography
            variant="subtitle1"
            sx={{ fontWeight: 800, fontFamily: "Montserrat", lineHeight: 1.2, color: "#fff" }}
          >
            {signal.symbol}
          </Typography>
          <Typography variant="caption" sx={{ color: "text.secondary", textTransform: "capitalize" }}>
            {signal.coin}
          </Typography>
        </Box>
        <Box sx={{ ml: "auto" }}>
          <Chip
            id={`signal-chip-${signal.coin}`}
            icon={<DirectionIcon sx={{ fontSize: "0.85rem !important" }} />}
            label={cfg.label}
            size="small"
            sx={{
              backgroundColor: cfg.bg,
              color: cfg.color,
              border: `1px solid ${cfg.color}55`,
              fontWeight: 700,
              fontFamily: "Montserrat",
              fontSize: "0.72rem",
            }}
          />
        </Box>
      </Box>

      {/* Confidence bar */}
      <ConfidenceBar value={signal.confidence} color={cfg.color} />

      {/* Rationale */}
      <Box
        sx={{
          mt: 1.5,
          p: 1.2,
          borderRadius: 2,
          backgroundColor: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.05)",
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: "rgba(255,255,255,0.65)",
            fontSize: "0.73rem",
            lineHeight: 1.5,
            fontStyle: "italic",
          }}
        >
          💬 {signal.rationale}
        </Typography>
      </Box>

      {/* Date */}
      <Typography
        variant="caption"
        sx={{ display: "block", mt: 1, color: "rgba(255,255,255,0.3)", fontSize: "0.65rem" }}
      >
        Signal date: {signal.date}
      </Typography>
    </Box>
  );
}

export default function SignalPanel() {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  const fetchSignals = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/signals`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSignals(data);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err.message);
      // Show mock signals if API unreachable
      setSignals(MOCK_SIGNALS);
      setLastRefresh(new Date());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchSignals]);

  return (
    <Box id="signal-panel" sx={{ p: { xs: 2, md: 4 } }}>
      {/* Panel header */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
        <Box>
          <Typography
            variant="h5"
            sx={{ fontFamily: "Montserrat", fontWeight: 800, color: "goldenrod", letterSpacing: 1 }}
          >
            🤖 LLM Signal Panel
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary", mt: 0.3 }}>
            Powered by Groq · llama3-70b · Refreshes every 60s
          </Typography>
        </Box>
        <Tooltip title="Refresh signals">
          <IconButton
            id="signal-refresh-btn"
            onClick={fetchSignals}
            disabled={loading}
            sx={{ color: "goldenrod" }}
          >
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Last refresh */}
      {lastRefresh && (
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.3)", display: "block", mb: 2 }}>
          Last updated: {lastRefresh.toLocaleTimeString()}
        </Typography>
      )}

      {/* Error banner */}
      {error && (
        <Box
          sx={{
            mb: 2, p: 1.5, borderRadius: 2,
            backgroundColor: "rgba(255,23,68,0.1)",
            border: "1px solid rgba(255,23,68,0.3)",
          }}
        >
          <Typography variant="caption" sx={{ color: "#ff6b6b" }}>
            ⚠️ API unreachable ({error}) — showing cached/mock signals. Run the FastAPI server to get live data.
          </Typography>
        </Box>
      )}

      {/* Loading */}
      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", py: 8 }}>
          <CircularProgress sx={{ color: "goldenrod" }} />
        </Box>
      ) : (
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", lg: "repeat(3, 1fr)" },
            gap: 2.5,
          }}
        >
          {signals.map((signal) => (
            <SignalCard key={signal.coin} signal={signal} />
          ))}
        </Box>
      )}

      {/* Research disclaimer */}
      <Box
        sx={{
          mt: 4, p: 2, borderRadius: 2,
          backgroundColor: "rgba(255,193,7,0.08)",
          border: "1px solid rgba(255,193,7,0.2)",
        }}
      >
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", lineHeight: 1.6 }}>
          📊 <strong style={{ color: "goldenrod" }}>Research System</strong> — These signals are generated by
          an LLM (Groq llama3-70b) from market data and are part of a backtested research project.
          They do not constitute financial advice and must not be used for live trading.
        </Typography>
      </Box>
    </Box>
  );
}

// Mock signals shown when API is unavailable
const MOCK_SIGNALS = [
  { coin: "bitcoin", symbol: "BTC", date: "2025-01-15", direction: "long", confidence: 0.72,
    rationale: "Strong upward momentum with increasing volume and positive sentiment crossover." },
  { coin: "ethereum", symbol: "ETH", date: "2025-01-15", direction: "hold", confidence: 0.41,
    rationale: "Mixed signals — high volatility with unclear directional bias. Awaiting confirmation." },
  { coin: "solana", symbol: "SOL", date: "2025-01-15", direction: "long", confidence: 0.81,
    rationale: "Bullish breakout above key resistance with strong volume confirmation." },
  { coin: "binancecoin", symbol: "BNB", date: "2025-01-15", direction: "short", confidence: 0.56,
    rationale: "Bearish divergence on 7-day momentum; sentiment turning negative amid exchange uncertainty." },
  { coin: "cardano", symbol: "ADA", date: "2025-01-15", direction: "hold", confidence: 0.34,
    rationale: "Insufficient signal strength. Price consolidating in tight range with low volume." },
];
