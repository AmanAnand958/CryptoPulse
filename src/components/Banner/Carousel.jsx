import axios from "axios";
import { useEffect, useState } from "react";
import AliceCarousel from "react-alice-carousel";
import { Link } from "react-router-dom";
import { TrendingCoins } from "../config/api";
import { CryptoState } from "../../CryptoContext";
import { numberWithCommas } from "../Coinstable";
import { styled } from '@mui/system';
import { Typography } from '@mui/material';

// Styled components
const CarouselContainer = styled('div')({
  height: '50%',
  display: 'flex',
  alignItems: 'center',
});

const CarouselItem = styled(Link)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  cursor: 'pointer',
  textTransform: 'uppercase',
  color: theme.palette.text.primary,
  textDecoration: 'none',
}));

const ItemImage = styled('img')({
  marginBottom: 20,
  marginTop: 40, // Add margin-top of 2px
  height: '60px', // Decrease the height of images
  width: 'auto', // Maintain aspect ratio
});

const PriceChange = styled('span')(({ profit }) => ({
  color: profit > 0 ? "rgb(14, 203, 129)" : "red",
  fontWeight: 500,
}));

const PriceText = styled('span')({
  fontSize: 18, // Decrease font size if needed
  fontWeight: 500,
});

const Carousel = () => {
  const [trending, setTrending] = useState([]);
  const { currency, symbol } = CryptoState();

  const fetchTrendingCoins = async () => {
    const { data } = await axios.get(TrendingCoins(currency));
    setTrending(data);
  };

  useEffect(() => {
    fetchTrendingCoins();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currency]);

  const items = trending.map((coin) => {
    const profit = coin?.price_change_percentage_24h >= 0;

    return (
      <CarouselItem to={`/coins/${coin.id}`} key={coin.id}>
        <ItemImage
          src={coin?.image}
          alt={coin.name}
        />
        <Typography variant="body2">
          {coin?.symbol}
          &nbsp;
          <PriceChange profit={profit}>
            {profit && "+"}
            {coin?.price_change_percentage_24h?.toFixed(2)}%
          </PriceChange>
        </Typography>
        <PriceText>
          {symbol} {numberWithCommas(coin?.current_price.toFixed(2))}
        </PriceText>
      </CarouselItem>
    );
  });

  const responsive = {
    0: {
      items: 2,
    },
    512: {
      items: 4,
    },
  };

  return (
    <CarouselContainer>
      <AliceCarousel
        mouseTracking
        infinite
        autoPlayInterval={1000}
        animationDuration={1500}
        disableDotsControls
        disableButtonsControls
        responsive={responsive}
        items={items}
        autoPlay
      />
    </CarouselContainer>
  );
};

export default Carousel;
