# Modeling Heating Oil and Crude Oil Fuel Price Relationships using Vector Error Correction Model... Markets don't move in isolation. If you've ever watched energy prices,
you've seen this: when crude oil prices rise, heating oil prices...

### Modeling Heating Oil and Crude Oil Fuel Price Relationships using **Vector Error Correction Model** (VECM) in Python
Markets don't move in isolation. If you've ever watched energy prices,
you've seen this: when crude oil prices rise, heating oil prices often
follow. These movements aren't always perfectly in sync in the short
term, but over time, they tend to settle into a consistent relationship.
This is the hallmark of cointegration.

We will build a Python pipeline to detect and model cointegration using
Crude Oil (CL=F) and Heating Oil (HO=F) futures. We'll use the Vector
Error Correction Model (VECM) to understand how these prices co-move in
both the short and long run --- and forecast where they're heading next.

Cointegration captures a long-term equilibrium relationship between two
or more non-stationary time series. Prices may diverge in the short term
but eventually return to a shared path. If two series are cointegrated,
you can't just model them separately --- you need to account for their
connection.

VECM blends short-run dynamics (like a VAR model) with long-run
equilibrium correction terms.

We'll use Yahoo Finance to pull daily futures prices for Crude Oil and
Heating Oil from 2015 to 2024.

```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data():
    symbols = ['HO=F', 'CL=F']  # Heating Oil and Crude Oil futures
    df = yf.download(symbols, start="2015-01-01", end="2024-12-31")['Close']
    df.dropna(inplace=True)
    df.columns = ['HeatingOil', 'CrudeOil']
    return df
df = fetch_data()
df.plot(title="Heating Oil vs Crude Oil Futures")
plt.ylabel("Price")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("cointegration_series.png")
plt.show()
```

We're working with non-stationary price levels --- ideal for
cointegration modeling.

### Step 2: Check for Stationarity
We apply the Augmented Dickey-Fuller (ADF) test to see if the raw and
differenced series are stationary.

```python
from statsmodels.tsa.stattools import adfuller

def adf_summary(df):
    for col in df.columns:
        result = adfuller(df[col])
        print(f"{col} ADF: {result[0]:.4f}, p={result[1]:.4f}")
        result_diff = adfuller(df[col].diff().dropna())
        print(f"{col} ΔADF: {result_diff[0]:.4f}, p={result_diff[1]:.4f}")
adf_summary(df)
```

Both price series are non-stationary in levels, but stationary in first
differences --- confirming they are I(1).

### Step 3: Test for Cointegration
We now use the Johansen test to determine if the series share a stable
long-run relationship.

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(df):
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    print("Trace Statistic:", result.lr1)
    print("Critical Values (90%, 95%, 99%):\n", result.cvt)
johansen_test(df)
```

The Johansen test results (Trace Statistic: \[50.96382923, 4.8986314\]
exceeds the 95 % critical values) provide strong statistical evidence
that Crude Oil and Heating Oil prices are cointegrated. This confirms
our earlier visual intuition: despite short-term fluctuations, these
fuel prices are tethered together by a long-run equilibrium.

### Step 4: Fit the VECM
We now fit the Vector Error Correction Model and inspect the dynamics.

```python
from statsmodels.tsa.vector_ar.vecm import VECM

vecm_model = VECM(df, k_ar_diff=1, coint_rank=1)
vecm_res = vecm_model.fit()
print(vecm_res.summary())
```


Loading coefficients show how each series adjusts back toward
equilibrium. Short-run coefficients show immediate reactions to recent
price changes.

As one would expect, we find Crude Oil acts as the anchor, with Heating
Oil adjusting more significantly.

### Step 5: Forecasting with VECM
Instead of using `predict()`, we use
`simulate_var()` to generate stable
multi-step forecasts.

```python
def forecast_vecm(res, df, steps=12):
    sim_forecast = res.simulate_var(steps=steps)
    future_idx = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
    forecast_df = pd.DataFrame(sim_forecast, columns=df.columns, index=future_idx)

df[-100:].plot(figsize=(10, 5), label='Historical')
    forecast_df.plot(ax=plt.gca(), style='--')
    plt.title("12-Month VECM Forecast")
    plt.savefig("vecm_forecast_corrected_simulated.png")
    plt.show()
forecast_vecm(vecm_res, df)
```

The resulting chart shows how the model expects these series to evolve,
always pulling back toward their long-run ratio.

### Step 6: Impulse Response and Variance Decomposition
We fit a short-run VAR on the differenced series to get insights into
**shock transmission**.

```python
from statsmodels.tsa.api import VAR

def run_var_irf_fevd(df_diff, lags=1, horizon=12):
    model = VAR(df_diff)
    res = model.fit(maxlags=lags)
    print(res.summary())
    irf = res.irf(horizon)
    irf.plot(orth=True)
    plt.suptitle("Impulse Response (Orthogonalized)")
    plt.savefig("var_irf_orthogonalized.png")
    plt.show()
    fevd = res.fevd(horizon)
    fevd.plot()
    plt.suptitle("Forecast Error Variance Decomposition (FEVD)")
    plt.savefig("var_fevd.png")
    plt.show()
run_var_irf_fevd(df.diff().dropna())
```


These plots show that Heating Oil reacts to Crude Oil shocks immediately
and strongly. Crude Oil forecasts are mostly driven by itself,
reinforcing that it leads the dynamic.

### Why This Matters
This model confirms what energy analysts often suspect: Heating Oil
prices are tethered to Crude, but they don't move identically. By
modeling cointegration explicitly, we capture that subtle balance of
independence and dependence.

You can use this approach for energy hedging, refinery profitability
modeling, and Long-short trading strategies.

This pipeline is extensible. You can change the inputs and see how other
things are related.

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def fetch_data(symbols, start="2015-01-01", end="2024-12-31"):
    """
    Fetches historical daily closing prices from Yahoo Finance for specified symbols.

    Args:
        symbols (list): List of ticker symbols to download (e.g., ['HO=F', 'CL=F']).
        start (str): Start date for the data in 'YYYY-MM-DD' format.
        end (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with columns ['HeatingOil', 'CrudeOil'] and date index.
    """
    df = yf.download(symbols, start=start, end=end)['Close'].dropna()
    df.columns = ['HeatingOil', 'CrudeOil']
    return df


def plot_series(df, filename="cointegration_series.png"):
    """
    Plots the input time series and saves the figure to a file.

    Args:
        df (pd.DataFrame): DataFrame with time series to plot.
        filename (str): Name of the output PNG file.
    """
    df.plot(title="Heating Oil vs Crude Oil Prices")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def adf_summary(df):
    """
    Performs Augmented Dickey-Fuller (ADF) tests on both levels and first differences
    of each column in the DataFrame and prints the results.

    Args:
        df (pd.DataFrame): DataFrame containing time series to test.
    """
    for col in df.columns:
        result = adfuller(df[col])
        print(f"{col} ADF: {result[0]:.4f}, p={result[1]:.4f}")
        result_diff = adfuller(df[col].diff().dropna())
        print(f"{col} ΔADF: {result_diff[0]:.4f}, p={result_diff[1]:.4f}")


def johansen_test(df):
    """
    Runs the Johansen cointegration test on the input data.

    Args:
        df (pd.DataFrame): DataFrame containing non-stationary I(1) time series.

    Returns:
        coint_johansen: The result object containing test statistics and eigenvectors.
    """
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    print("Trace Statistic:", result.lr1)
    print("Critical Values (90%, 95%, 99%):\n", result.cvt)
    return result


def fit_vecm(df, lags=1, rank=1):
    """
    Fits a Vector Error Correction Model (VECM) to the data.

    Args:
        df (pd.DataFrame): Time series data to model.
        lags (int): Number of lags to use in the VECM.
        rank (int): Cointegration rank determined from Johansen test.

    Returns:
        VECMResults: Fitted VECM model results object.
    """
    model = VECM(df, k_ar_diff=lags, coint_rank=rank)
    res = model.fit()
    print(res.summary())
    return res


def forecast_vecm(res, df, steps=12):
    """
    Forecasts future values using the fitted VECM and plots results.

    Args:
        res (VECMResults): Fitted VECM results object.
        df (pd.DataFrame): Original time series data.
        steps (int): Number of periods to forecast forward.
    """
    forecast = res.predict(steps=steps)
    future_idx = pd.date_range(df.index[-1], periods=steps, freq='ME')
    forecast_df = pd.DataFrame(forecast, columns=df.columns, index=future_idx)
    
    df[-100:].plot(figsize=(10, 5), label='Historical')
    forecast_df.plot(ax=plt.gca(), style='--')
    plt.title("12-Month VECM Forecast")
    plt.savefig("vecm_forecast.png")
    plt.show()


def run_var_irf_fevd(df_diff, lags=1, horizon=12):
    """
    Fits a VAR model on differenced data, plots impulse response functions (IRFs)
    and forecast error variance decomposition (FEVD).

    Args:
        df_diff (pd.DataFrame): First-differenced time series data.
        lags (int): Number of lags for the VAR model.
        horizon (int): Forecast horizon for IRF and FEVD plots.
    """
    model = VAR(df_diff)
    res = model.fit(maxlags=lags)
    print(res.summary())

    irf = res.irf(horizon)
    irf.plot(orth=False)
    plt.suptitle("Impulse Response (Standard)")
    plt.savefig("var_irf_standard.png")
    plt.show()

    irf.plot(orth=True)
    plt.suptitle("Impulse Response (Orthogonalized)")
    plt.savefig("var_irf_orthogonalized.png")
    plt.show()

    fevd = res.fevd(horizon)
    fevd.plot()
    plt.suptitle("Forecast Error Variance Decomposition (FEVD)")
    plt.savefig("var_fevd.png")
    plt.show()


def main():
    """
    Runs the full cointegration analysis pipeline:
    - Downloads data
    - Performs stationarity and cointegration tests
    - Fits and forecasts VECM
    - Plots IRFs and FEVD from a VAR on differenced data
    """
    symbols = ['HO=F', 'CL=F']  # Heating Oil and Crude Oil
    df = fetch_data(symbols)
    plot_series(df)
    adf_summary(df)

    johansen_test(df)
    vecm_res = fit_vecm(df, lags=1, rank=1)
    forecast_vecm(vecm_res, df)

    df_diff = df.diff().dropna()
    run_var_irf_fevd(df_diff, lags=1, horizon=12)


if __name__ == "__main__":
    main()
```
::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[April 25, 2025](https://medium.com/p/d2af5214fa31).

[Canonical
link](https://medium.com/@kyle-t-jones/modeling-heating-oil-and-crude-oil-fuel-price-relationships-using-vector-error-correction-model-d2af5214fa31)

Exported from [Medium](https://medium.com) on November 10, 2025.
