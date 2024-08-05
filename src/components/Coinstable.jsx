import React, { useEffect, useState } from "react";
import { styled } from "@mui/system";
import Pagination from "@mui/material/Pagination";
import {
  Container,
  createTheme,
  TableCell,
  LinearProgress,
  ThemeProvider,
  Typography,
  TextField,
  TableBody,
  TableRow,
  TableHead,
  TableContainer,
  Table,
  Paper,
  useTheme,
} from "@mui/material";
import axios from "axios";
import { CoinList } from "./config/api";
import { useNavigate } from 'react-router-dom';
import { CryptoState } from "../CryptoContext";

export function numberWithCommas(x) {
  return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Styled components
const StyledTableRow = styled(TableRow)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  cursor: "pointer",
  "&:hover": {
    backgroundColor: theme.palette.action.hover,
  },
  fontFamily: theme.typography.fontFamily,
}));

const StyledPagination = styled(Pagination)(({ theme }) => ({
  "& .MuiPaginationItem-root": {
    color: theme.palette.text.primary,
  },
}));

const CoinImage = styled('img')(({ theme }) => ({
  marginBottom: theme.spacing(1),
  height: '40px', // Decrease the height of the image
}));

export default function CoinsTable() {
  const [coins, setCoins] = useState([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const { currency, symbol } = CryptoState();
  const history = useNavigate();
  const theme = useTheme(); // Access the current theme

  const darkTheme = createTheme({
    palette: {
      mode: theme.palette.mode, // Use the current theme mode (light/dark)
      primary: {
        main: "#fff",
      },
      secondary: {
        main: "#EEBC1D",
      },
      background: {
        paper: theme.palette.mode === "dark" ? "#16171a" : "#fff",
      },
      action: {
        hover: theme.palette.mode === "dark" ? "#131111" : "#f0f0f0",
      },
      text: {
        primary: theme.palette.mode === "dark" ? "#fff" : "#000",
        secondary: theme.palette.mode === "dark" ? "#aaa" : "#555",
      },
    },
    typography: {
      fontFamily: "Montserrat",
    },
  });

  const fetchCoins = async () => {
    setLoading(true);
    const { data } = await axios.get(CoinList(currency));
    setCoins(data);
    setLoading(false);
  };

  useEffect(() => {
    fetchCoins();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currency]);

  const handleSearch = () => {
    return coins.filter(
      (coin) =>
        coin.name.toLowerCase().includes(search.toLowerCase()) ||
        coin.symbol.toLowerCase().includes(search.toLowerCase())
    );
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <Container sx={{ textAlign: "center" }}>
        <Typography
          variant="h4"
          sx={{ margin: 2, fontFamily: "Montserrat" }}
        >
          CRYPTO CURRENCY PRICES
        </Typography>
        <TextField
          label="Search For a Crypto Currency.."
          variant="outlined"
          sx={{ marginBottom: 2, width: "100%" }}
          onChange={(e) => setSearch(e.target.value)}
        />
        <TableContainer component={Paper}>
          {loading ? (
            <LinearProgress sx={{ backgroundColor: "gold" }} />
          ) : (
            <Table aria-label="simple table">
              <TableHead sx={{ backgroundColor: darkTheme.palette.secondary.main }}>
                <TableRow>
                  {["Coin", "Price", "24h Change", "Market Cap"].map((head) => (
                    <TableCell
                      sx={{
                        color: darkTheme.palette.text.primary,
                        fontWeight: "700",
                        fontFamily: "Montserrat",
                      }}
                      key={head}
                      align={head === "Coin" ? "left" : "right"}
                    >
                      {head}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>

              <TableBody>
                {handleSearch()
                  .slice((page - 1) * 10, (page - 1) * 10 + 10)
                  .map((row) => {
                    const profit = row.price_change_percentage_24h > 0;
                    return (
                      <StyledTableRow
                        onClick={() => history(`/coins/${row.id}`)}
                        key={row.id}
                      >
                        <TableCell
                          component="th"
                          scope="row"
                          sx={{ display: "flex", gap: 2 }}
                        >
                          <CoinImage
                            src={row?.image}
                            alt={row.name}
                          />
                          <div
                            style={{ display: "flex", flexDirection: "column" }}
                          >
                            <span
                              style={{
                                textTransform: "uppercase",
                                fontSize: 22,
                              }}
                            >
                              {row.symbol}
                            </span>
                            <span style={{ color: darkTheme.palette.text.secondary }}>
                              {row.name}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell align="right">
                          {symbol} {numberWithCommas(row.current_price.toFixed(2))}
                        </TableCell>
                        <TableCell
                          align="right"
                          sx={{
                            color: profit ? "rgb(14, 203, 129)" : "red",
                            fontWeight: 500,
                          }}
                        >
                          {profit && "+"}
                          {row.price_change_percentage_24h.toFixed(2)}%
                        </TableCell>
                        <TableCell align="right">
                          {symbol} {numberWithCommas((row.market_cap / 1e6).toFixed(0))} M
                        </TableCell>
                      </StyledTableRow>
                    );
                  })}
              </TableBody>
            </Table>
          )}
        </TableContainer>

        <StyledPagination
          count={Math.ceil(handleSearch().length / 10)}
          sx={{
            padding: 2,
            width: "100%",
            display: "flex",
            justifyContent: "center",
          }}
          onChange={(_, value) => {
            setPage(value);
            window.scroll(0, 450);
          }}
        />
      </Container>
    </ThemeProvider>
  );
}
