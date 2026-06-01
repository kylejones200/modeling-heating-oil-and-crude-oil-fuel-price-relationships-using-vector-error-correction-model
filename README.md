# Modeling Heating Oil and Crude Oil Price Relationships Using VECM

This project demonstrates Vector Error Correction Model (VECM) for analyzing cointegrated time series.

## Business context

Markets don't move in isolation. If you've ever watched energy prices, you've seen this: when crude oil prices rise, heating oil prices often follow. These movements aren't always perfectly in sync in the short term, but over time, they tend to settle into a consistent relationship. This is the hallmark of cointegration.

We will build a Python pipeline to detect and model cointegration using Crude Oil (CL=F) and Heating Oil (HO=F) futures. We'll use the Vector Error Correction Model (VECM) to understand how these prices co-move in both the short and long run --- and forecast where they're heading next.

Cointegration captures a long-term equilibrium relationship between two or more non-stationary time series. Prices may diverge in the short term but eventually return to a shared path. If two series are cointegrated, you can't just model them separately --- you need to account for their connection.

## Article

Medium article: [Modeling Heating Oil and Crude Oil Price Relationships Using VECM](https://medium.com/@kylejones_47003/modeling-heating-oil-and-crude-oil-fuel-price-relationships-using-vector-error-correction-model-d2af5214fa31)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # VECM modeling functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data symbols and date range
- VECM model parameters (lags, cointegration rank)
- Forecast horizon
- Output settings

## VECM Model

Vector Error Correction Model:
- Cointegration: Long-run equilibrium relationship
- Error Correction: Short-run adjustment mechanism
- Forecasting: Multi-step ahead predictions
- Stationarity: Requires I(1) series with cointegration

## Caveats

- Requires internet connection to fetch data from Yahoo Finance.
- Data availability depends on Yahoo Finance API.
- VECM requires cointegrated I(1) series.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).