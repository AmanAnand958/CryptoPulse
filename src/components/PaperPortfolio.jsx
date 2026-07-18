import React, { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  CircularProgress,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  Alert,
} from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const COINS = [
  { id: "bitcoin", label: "BTC", color: "#F7931A" },
  { id: "ethereum", label: "ETH", color: "#627EEA" },
  { id: "solana", label: "SOL", color: "#9945FF" },
  { id: "binancecoin", label: "BNB", color: "#F3BA2F" },
  { id: "cardano", label: "ADA", color: "#0033AD" },
];

const DIRECTION_COLORS = {
  long: "#00e676",
  short: "#ff1744",
  hold: "#78909c",
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const dirColor = DIRECTION_COLORS[d.direction] || "#78909c";
  return (
    <Box
      sx={{
        background: "rgba(20,20,30,0.95)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 2,
        p: 1.5,
        minWidth: 180,
      }}
    >
      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", display: "block", mb: 0.5 }}>
        {d.date}
      </Typography>
      <Typography variant="body2" sx={{ color: "#fff", fontWeight: 700 }}>
        ${Number(d.portfolio_value).toFixed(2)}
      </Typography>
      {d.direction && (
        <Chip
          label={d.direction.toUpperCase()}
          size="small"
          sx={{
            mt: 0.5,
            backgroundColor: `${dirColor}22`,
            color: dirColor,
            border: `1px solid ${dirColor}55`,
            fontSize: "0.65rem",
            height: 18,
          }}
        />
      )}
    </Box>
  );
}

function formatDate(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function PortfolioStats({ data }) {
  if (!data || data.length < 2) return null;
  const initial = data[0]?.portfolio_value || 10000;
  const final = data[data.length - 1]?.portfolio_value || 10000;
  const totalReturn = (final / initial - 1) * 100;
  const peak = Math.max(...data.map((d) => d.portfolio_value));
  const trough = Math.min(...data.map((d) => d.portfolio_value));
  const maxDD = ((peak - trough) / peak) * 100;

  const statStyle = { textAlign: "center", flex: 1 };
  const labelStyle = { color: "rgba(255,255,255,0.4)", fontSize: "0.7rem", display: "block", mb: 0.3 };

  return (
    <Box
      sx={{
        display: "flex",
        gap: 1,
        p: 2,
        borderRadius: 2,
        backgroundColor: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.06)",
        mb: 2,
        flexWrap: "wrap",
      }}
    >
      <Box sx={statStyle}>
        <Typography variant="caption" sx={labelStyle}>Starting Value</Typography>
        <Typography variant="body2" sx={{ color: "#fff", fontWeight: 700 }}>${initial.toFixed(0)}</Typography>
      </Box>
      <Box sx={statStyle}>
        <Typography variant="caption" sx={labelStyle}>Current Value</Typography>
        <Typography variant="body2" sx={{ color: "#fff", fontWeight: 700 }}>${final.toFixed(0)}</Typography>
      </Box>
      <Box sx={statStyle}>
        <Typography variant="caption" sx={labelStyle}>Total Return</Typography>
        <Typography variant="body2" sx={{ color: totalReturn >= 0 ? "#00e676" : "#ff1744", fontWeight: 700 }}>
          {totalReturn >= 0 ? "+" : ""}{totalReturn.toFixed(2)}%
        </Typography>
      </Box>
      <Box sx={statStyle}>
        <Typography variant="caption" sx={labelStyle}>Max Drawdown</Typography>
        <Typography variant="body2" sx={{ color: "#ff9800", fontWeight: 700 }}>{maxDD.toFixed(2)}%</Typography>
      </Box>
    </Box>
  );
}

export default function PaperPortfolio() {
  const [selectedCoin, setSelectedCoin] = useState("bitcoin");
  const [portfolioData, setPortfolioData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPortfolio = useCallback(async (coin) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/portfolio?coin=${coin}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const data = json.data || json;
      // Sample every 2nd point for chart performance
      const sampled = Array.isArray(data) ? data.filter((_, i) => i % 2 === 0) : [];
      setPortfolioData(sampled);
    } catch (err) {
      setError(err.message);
      setPortfolioData(generateMockPortfolio());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPortfolio(selectedCoin);
  }, [selectedCoin, fetchPortfolio]);

  const initialValue = portfolioData[0]?.portfolio_value || 10000;
  const coinConfig = COINS.find((c) => c.id === selectedCoin) || COINS[0];

  return (
    <Box id="paper-portfolio-panel" sx={{ p: { xs: 2, md: 4 } }}>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", mb: 2, flexWrap: "wrap", gap: 2 }}>
        <Box>
          <Typography
            variant="h5"
            sx={{ fontFamily: "Montserrat", fontWeight: 800, color: "goldenrod", letterSpacing: 1 }}
          >
            📋 Paper Portfolio
          </Typography>
          <Box sx={{ display: "inline-block", mt: 0.8 }}>
            <Box
              id="paper-trading-badge"
              sx={{
                px: 2, py: 0.5, borderRadius: 20,
                background: "linear-gradient(90deg, rgba(255,23,68,0.2), rgba(255,152,0,0.2))",
                border: "1.5px solid rgba(255,152,0,0.6)",
                display: "inline-flex",
                alignItems: "center",
                gap: 0.8,
              }}
            >
              <Box
                sx={{
                  width: 8, height: 8, borderRadius: "50%",
                  backgroundColor: "#ff9800",
                  animation: "pulse 1.5s infinite",
                  "@keyframes pulse": {
                    "0%, 100%": { opacity: 1 },
                    "50%": { opacity: 0.4 },
                  },
                }}
              />
              <Typography
                variant="caption"
                sx={{ color: "#ff9800", fontWeight: 800, fontFamily: "Montserrat", letterSpacing: 0.5 }}
              >
                PAPER TRADING — NOT REAL FUNDS
              </Typography>
            </Box>
          </Box>
        </Box>

        {/* Coin selector */}
        <ToggleButtonGroup
          id="portfolio-coin-selector"
          value={selectedCoin}
          exclusive
          onChange={(_, val) => val && setSelectedCoin(val)}
          size="small"
        >
          {COINS.map((c) => (
            <ToggleButton
              key={c.id}
              value={c.id}
              id={`portfolio-btn-${c.id}`}
              sx={{
                color: selectedCoin === c.id ? c.color : "rgba(255,255,255,0.4)",
                borderColor: "rgba(255,255,255,0.1)",
                fontSize: "0.72rem",
                fontWeight: 700,
                fontFamily: "Montserrat",
                "&.Mui-selected": {
                  backgroundColor: `${c.color}20`,
                  color: c.color,
                  borderColor: `${c.color}55`,
                },
              }}
            >
              {c.label}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>

      {/* Error banner */}
      {error && (
        <Alert severity="warning" sx={{ mb: 2, backgroundColor: "rgba(255,152,0,0.1)", color: "#ff9800" }}>
          API unreachable — showing simulated demo data. Start FastAPI server for live results.
        </Alert>
      )}

      {/* Stats */}
      <PortfolioStats data={portfolioData} />

      {/* Chart */}
      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", py: 10 }}>
          <CircularProgress sx={{ color: coinConfig.color }} />
        </Box>
      ) : (
        <Box
          id="portfolio-chart"
          sx={{
            background: "rgba(255,255,255,0.02)",
            borderRadius: 3,
            border: "1px solid rgba(255,255,255,0.06)",
            p: 2,
          }}
        >
          <Typography variant="subtitle2" sx={{ color: "rgba(255,255,255,0.5)", mb: 1.5, fontFamily: "Montserrat" }}>
            {coinConfig.label} — Simulated Portfolio Value (LinUCB Bandit Policy)
          </Typography>
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={portfolioData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              />
              <YAxis
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine
                y={initialValue}
                stroke="rgba(255,255,255,0.2)"
                strokeDasharray="4 4"
                label={{ value: "Start", fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
              />
              <Line
                type="monotone"
                dataKey="portfolio_value"
                stroke={coinConfig.color}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5, fill: coinConfig.color, stroke: "#fff", strokeWidth: 1.5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* Disclaimer */}
      <Box
        sx={{
          mt: 3, p: 2, borderRadius: 2,
          backgroundColor: "rgba(255,23,68,0.06)",
          border: "1px solid rgba(255,23,68,0.2)",
        }}
      >
        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.5)", lineHeight: 1.6 }}>
          ⚠️ <strong style={{ color: "#ff6b6b" }}>Paper Trading Only.</strong> All portfolio values shown
          are simulated using historical backtested data and the LinUCB bandit execution policy.
          No real funds are involved. Past backtest performance does not indicate future results.
          This is a research tool, not financial advice.
        </Typography>
      </Box>
    </Box>
  );
}

function generateMockPortfolio() {
  let value = 10000;
  const data = [];
  for (let i = 0; i < 90; i++) {
    const r = (Math.random() - 0.48) * 0.04;
    value = Math.max(value * (1 + r), 1000);
    const d = new Date("2024-06-01");
    d.setDate(d.getDate() + i);
    data.push({
      date: d.toISOString().slice(0, 10),
      portfolio_value: parseFloat(value.toFixed(2)),
      direction: ["long", "short", "hold"][i % 3],
      position_multiplier: [-1, -0.5, 0, 0.5, 1][i % 5],
    });
  }
  return data;
}
