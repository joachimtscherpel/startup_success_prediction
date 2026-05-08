import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import requests
import ast

class FXConverter(BaseEstimator, TransformerMixin):
    """
    Cacheable FX converter: uses preloaded FX rates from disk.
    Fully deterministic for joblib.Memory caching.
    """
    def __init__(self, base_currency="EUR", target_currency="USD", cache_file="data/fx_rates.pkl"):
        # DO NOT mutate constructor args
        self.base_currency = base_currency
        self.target_currency = target_currency
        self.cache_file = cache_file

        # internal attributes
        self.fx_rates_ = None
        self.base_currency_ = None
        self.target_currency_ = None

    def fit(self, X=None, y=None):
        # store normalized currencies internally
        self.base_currency_ = str(self.base_currency).upper()
        self.target_currency_ = str(self.target_currency).upper()

        # Load cached FX rates
        self._load_cache()
        if self.fx_rates_ is None:
            self.fx_rates_ = {}

        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        return X

    def convert_to_usd(self, amount, currency, date):
        if currency is None or pd.isna(amount) or pd.isna(date):
            return float('nan')

        currency = str(currency).upper()
        date = pd.to_datetime(date)

        # already in target currency
        if currency == self.target_currency_:
            return float(amount)

        series = self.fx_rates_.get(f"{self.base_currency_}{currency}")
        if series is None or series.empty:
            return float('nan')
        rate = series.asof(date)
        if pd.isna(rate):
            return float('nan')
        return float(amount * rate)

    def _load_cache(self):
        try:
            self.fx_rates_ = pd.read_pickle(self.cache_file)
        except FileNotFoundError:
            self.fx_rates_ = {}

class FXRateDownloader:
    """
    Handles downloading FX rates from API and saving to cache.
    Not part of pipeline; used offline to update the cache.
    """
    def __init__(self, base_currency="EUR", target_currency="USD", cache_file="data/fx_rates.pkl", timeout=30):
        self.base_currency = base_currency.upper()
        self.target_currency = target_currency.upper()
        self.cache_file = cache_file
        self.timeout = timeout

    def download_for_df(self, df, date_col="round_dates", currency_col="round_currencies"):
        """
        Download missing FX rates for all currencies/dates in the dataframe.
        Saves the rates to cache_file.
        """
        start, end, currencies = self._extract_required_data(df, date_col, currency_col)
        if self.target_currency not in currencies:
            currencies.append(self.target_currency)

        fx_rates = {}
        for currency in currencies:
            key = f"{self.base_currency}{currency}"
            series = self._download_series(self.base_currency, currency, start, end)
            if not series.empty:
                fx_rates[key] = series.ffill()

        # Save to disk
        try:
            pd.to_pickle(fx_rates, self.cache_file)
        except Exception as e:
            warnings.warn(f"Failed to save FX cache: {e}")

    # ---- Private helpers ----
    def _extract_required_data(self, df, date_col="round_dates", currency_col="round_currencies"):
        dates_series = df[date_col].dropna()
        parsed_dates = dates_series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).explode()
        parsed_dates = pd.to_datetime(parsed_dates, errors="coerce").dropna()
        required_start = parsed_dates.min()
        required_end = parsed_dates.max()

        currency_series = df[currency_col].dropna()
        parsed_currencies = currency_series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).explode()
        parsed_currencies = parsed_currencies.astype(str).str.upper().str.strip()
        invalid = {"USD", "N/A", "NAN", "NONE", "", self.base_currency}
        required_currencies = sorted(set(parsed_currencies) - invalid)
        return required_start, required_end, required_currencies

    def _download_series(self, base, target, start, end):
        start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end).strftime("%Y-%m-%d")
        url = f"https://api.frankfurter.app/{start_str}..{end_str}?from={base}&to={target}"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "rates" not in data:
                return pd.Series(dtype=float)
            s = pd.Series(
                {pd.to_datetime(date): values[target] for date, values in data["rates"].items()}
            ).sort_index()
            return s.ffill() if not s.empty else s
        except Exception:
            return pd.Series(dtype=float)