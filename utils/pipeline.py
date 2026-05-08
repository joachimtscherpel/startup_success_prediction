from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from pathlib import Path
from joblib import Memory
import warnings
import numpy as np
import pandas as pd
from itertools import zip_longest
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor, XGBClassifier
from xgboost.callback import TrainingCallback
from joblib import Memory
from sklearn.base import clone
from typing import Iterator

# ============================================
# MODULE-LEVEL FILE CACHES
# One entry per worker process — avoids repeated disk reads
# across CV fits for the same embedding/score file.
# ============================================
_SCORE_CACHE: dict = {}       # path → raw DataFrame
_EMBEDDING_CACHE: dict = {}   # path → raw DataFrame  
_PCA_CACHE: dict = {}         # (path, n_components) → (fitted PCA, reduced DataFrame)

# ============================================
# PREPROCESSING PIPELINE COMPONENTS
# ============================================

class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, step_name):
        self.step_name = step_name
        self.output_ = None
    
    def fit(self, X, y=None):
        self._is_fitted = True
        return self
    
    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        
        self.output_ = X.copy()  # store a copy
        print(f"   - debugger output shape:", X.shape)
        return X

class DataLoader(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible dataloader that can be used in pipelines.
    Allows loading only the first `n_rows` for testing purposes.
    """
    def __init__(self, file_path=None, n_rows=None):        
        self.file_path = file_path
        self.n_rows = n_rows  # Limit rows for testing
        self.data = None
    
    def fit(self, X=None, y=None):
        # In a dataloader, fit might load the data
        if self.file_path:
            self._load_from_file()
        self._is_fitted = True
        return self
    
    def transform(self, X=None):
        # Pass-through for pipeline compatibility
        check_is_fitted(self, "_is_fitted")
        if X is not None:
            return X
        return self.data
    
    def _load_from_file(self):
        """Load data from file, optionally limiting to first n_rows, and ensure 'ID' is string."""
        file_ext = Path(self.file_path).suffix.lower()
        
        # Load data
        if file_ext == '.csv':
            # Ensure 'ID' is read as string to preserve leading zeros
            self.data = pd.read_csv(self.file_path, nrows=self.n_rows, dtype={'ID': str})
        elif file_ext == '.parquet':
            self.data = pd.read_parquet(self.file_path)
            if self.n_rows:
                self.data = self.data.head(self.n_rows)
        elif file_ext == '.pkl':
            self.data = pd.read_pickle(self.file_path)
            if self.n_rows:
                self.data = self.data.head(self.n_rows)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: .csv, .parquet, .pkl")

        # Ensure 'ID' exists and is string
        if 'ID' not in self.data.columns:
            if self.data.index.name == 'ID':
                self.data.reset_index(inplace=True)
            else:
                raise KeyError("No 'ID' column found in the data.")
        
        self.data['ID'] = self.data['ID'].astype(str)
        
class ColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, rename_dict):
        self.rename_dict = rename_dict
    
    def fit(self, X, y=None):
        self._is_fitted = True
        return self
    
    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        return X.rename(columns=self.rename_dict)

class DuplicateDropper(BaseEstimator, TransformerMixin):
    """
    Drops duplicate rows based on a specified column, keeping the row with the fewest missing values.
    """
    def __init__(self, duplicate_col):
        self.duplicate_col = duplicate_col
        
    def fit(self, X, y=None):
        self._is_fitted = True
        return self
    
    def transform(self, X): 
        check_is_fitted(self, "_is_fitted")
        
        if isinstance(X, pd.DataFrame):         
            # Make a copy to avoid modifying original
            X = X.copy()
            
            # Calculate missing value count per row
            missing_counts = X.isnull().sum(axis=1)
            
            # Add missing count as temporary column
            X['_missing_count'] = missing_counts
            
            # Sort by duplicate_col and missing count (ascending = fewer missing)
            X_sorted = X.sort_values([self.duplicate_col, '_missing_count'])
            
            # Drop duplicates keeping first (which has fewest missing)
            X_deduped = X_sorted.drop_duplicates(subset=[self.duplicate_col], keep='first')
            
            # Remove temporary column
            X_deduped = X_deduped.drop(columns=['_missing_count'])
            return X_deduped
        
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None): 
        self.cols_to_drop = cols_to_drop
    
    def fit(self, X, y=None):
        self._is_fitted = True 
        return self
    
    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        
        X = X.copy()
        if self.cols_to_drop and isinstance(X, pd.DataFrame):
            return X.drop(columns=self.cols_to_drop, errors='ignore')
        return X
    
class Prefilterer(BaseEstimator, TransformerMixin):
    """Filter rows where column values contain specified keywords."""
    
    def __init__(self, filter_map=None):
        """
        filter_map: dict
            Keys are column names, values are list of keywords to filter in that column.
            Example: {'col1': ['a', 'b'], 'col2': ['x']}
        """
        self.filter_map = filter_map or {}
    
    def fit(self, X, y=None):
        self._is_fitted = True
        return self
    
    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        
        if not isinstance(X, pd.DataFrame) or not self.filter_map:
            return X
        
        X = X.copy()
        before = len(X)
        
        # Start with all True (keep all rows)
        mask = pd.Series([True] * len(X), index=X.index)
        
        for col, keywords in self.filter_map.items():
            if col in X.columns and keywords:
                for keyword in keywords:
                    contains_keyword = X[col].astype(str).str.contains(keyword, case=False, na=False)
                    mask = mask & ~contains_keyword  # False where keyword found
        
        X_filtered = X[mask]
        return X_filtered
    
class ListParser(BaseEstimator, TransformerMixin):
    """
    Parses various list formats and date columns in a DataFrame.
    Column types are hardcoded based on the dataset structure.
    """
    
    MONTH_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    def __init__(self):
        # Hardcoded column types
        self.list_columns = [
            "team_dealroom", "team_editorial", "client_focus",
            "revenue_model", "industries", "sub_industries", "round_types",
            "round_currencies", "founders", "founders_genders", "serial_founder",
            "founders_first_degree", "top_past_founder", "founder_top_university",
            "founders_first_company", "founders_strength", "tags", "sdgs",
            'technologies', 'income_streams', 'tech_stack',
        ]
        
        self.numeric_list_columns = [
            "valuation_values_usd", "employee_values", "round_amounts",
            "founders_first_degree_year", "founders_total_funding",
            "founders_years_education"
        ]
        
        self.slash_date_columns = ["valuation_date"]
        self.dash_date_list_columns = ["valuation_dates"]
        self.comma_date_columns = ["launch_date", "closing_date"]
        self.standard_date_list_columns = ["employee_dates"]
        self.slash_date_list_columns = ["round_dates"]  # Added missing column
        self.pluplus_list_columns = ["round_investors"]
        self.comma_list_columns = [
            "founders_backgrounds", "founders_universities", "founders_company_experience"
        ]
        self.numeric_columns = ["valuation_usd"]
        self.year_columns = ["seed_year"]
    
    def fit(self, X, y=None):
        # Auto-detect yearly columns and add to numeric_list_columns
        if isinstance(X, pd.DataFrame):
            yearly_cols = [col for col in X.columns if "yearly" in col.lower()]
            self.numeric_list_columns = list(set(self.numeric_list_columns + yearly_cols))
        self._is_fitted = True
        return self
    
    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        
        if not isinstance(X, pd.DataFrame):
            return X
        
        X = X.copy()
        
        # Parse semicolon-separated lists
        for col in self.list_columns:
            if col in X.columns:
                X[col] = X[col].apply(self._parse_list)
        
        # Parse numeric lists
        for col in self.numeric_list_columns:
            if col in X.columns:
                X[col] = (X[col]
                         .apply(self._parse_list)
                         .apply(lambda lst: [self._to_float_or_nan(x) for x in lst]))
        
        # Parse slash dates (MMM/YYYY)
        for col in self.slash_date_columns:
            if col in X.columns:
                X[col] = X[col].apply(self._parse_slash_date)
        
        # Parse dash dates (MM-YYYY)
        for col in self.dash_date_list_columns:
            if col in X.columns:
                X[col] = (X[col]
                         .apply(self._parse_list)
                         .apply(lambda lst: [self._parse_dash_date(x) for x in lst]))
        
        # Parse standard dates
        for col in self.comma_date_columns:
            if col in X.columns:
                X[col] = X[col].apply(self._parse_comma_date)
        
        # Parse comma dates (YYYY, Month)
        for col in self.standard_date_list_columns:
            if col in X.columns:
                X[col] = (X[col]
                         .apply(self._parse_list)
                         .apply(lambda lst: [self._parse_standard_date(x) for x in lst]))
        
        # Parse slash date lists (list of MMM/YYYY dates)
        for col in self.slash_date_list_columns:
            if col in X.columns:
                X[col] = (X[col]
                         .apply(self._parse_list)
                         .apply(lambda lst: [self._parse_slash_date(x) for x in lst]))
        
        # Parse pluplus-separated lists (with ++)
        for col in self.pluplus_list_columns:
            if col in X.columns:
                X[col] = X[col].apply(self._parse_pluplus_semicolon_list)
        
        # Parse comma-separated inner lists
        for col in self.comma_list_columns:
            if col in X.columns:
                X[col] = X[col].apply(self._parse_comma_semicolon_list)
        
        # Convert to numeric
        for col in self.numeric_columns:
            if col in X.columns:
                X[col] = X[col].apply(self._to_float_or_nan)
        
        # Convert to integer years
        for col in self.year_columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').astype('Int64')
        
        total_columns = len(
            self.list_columns
            + self.numeric_list_columns
            + self.slash_date_columns
            + self.dash_date_list_columns
            + self.comma_date_columns
            + self.standard_date_list_columns
            + self.slash_date_list_columns
            + self.pluplus_list_columns
            + self.comma_list_columns
            + self.numeric_columns
            + self.year_columns
        )
        return X
    
    # ==================== HELPER METHODS ====================
    
    def _recursive_list_to_tuple(self, obj):
        """Recursively convert lists to tuples (works for nested lists)"""
        if isinstance(obj, list):
            return tuple(self._recursive_list_to_tuple(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: self._recursive_list_to_tuple(v) for k, v in obj.items()}
        else:
            return obj
    
    def _parse_list(self, s):
        """Parse semicolon-separated string into list"""
        if pd.isna(s) or s == "":
            return []
        return [x.strip() if x.strip() else "n/a" for x in str(s).split(";")]
    
    def _parse_standard_date(self, d): 
        return pd.to_datetime(d, errors="coerce")

    def _parse_dash_date(self, d):
        """Parse MM-YYYY format or just YYYY"""
        if pd.isna(d) or str(d).strip() == "":
            return pd.NaT
        
        d_str = str(d).strip()
        
        # Year-only
        if d_str.isdigit() and len(d_str) == 4:
            return pd.Timestamp(year=int(d_str), month=1, day=1)
        
        # MM-YYYY
        try:
            m, y = d_str.split("-")
            return pd.Timestamp(year=int(y), month=int(m), day=1)
        except:
            return pd.NaT

    def _parse_slash_date(self, d):
        """Parse MMM/YYYY format or just YYYY"""
        if pd.isna(d) or str(d).strip() == "":
            return pd.NaT
        
        d_str = str(d).strip()
        
        # Year-only
        if d_str.isdigit() and len(d_str) == 4:
            return pd.Timestamp(year=int(d_str), month=1, day=1)
        
        # MMM/YYYY
        try:
            m, y = d_str.split("/")
            month_num = self.MONTH_MAP.get(m[:3].lower(), 1)
            return pd.Timestamp(year=int(y), month=month_num, day=1)
        except:
            return pd.NaT

    def _parse_comma_date(self, d):
        """Parse YYYY, Month format or just YYYY"""
        if pd.isna(d) or str(d).strip() == "":
            return pd.NaT
        
        d_str = str(d).strip()
        
        # Year-only
        if d_str.isdigit() and len(d_str) == 4:
            return pd.Timestamp(year=int(d_str), month=1, day=1)
        
        # YYYY, Month
        try:
            year_str, month_str = map(str.strip, d_str.split(","))
            month_num = self.MONTH_MAP.get(month_str[:3].lower(), 1)
            return pd.Timestamp(year=int(year_str), month=month_num, day=1)
        except:
            return pd.NaT
    
    def _to_float_or_nan(self, x):
        """Convert value to float, handling ranges and n/a"""
        if pd.isna(x):
            return np.nan
        x_str = str(x).strip()
        if x_str.lower() in ["n/a", ""]:
            return np.nan
        if "-" in x_str:
            try:
                low, high = map(float, x_str.split("-"))
                return (low + high) / 2
            except:
                return np.nan
        try:
            return float(x_str)
        except:
            return np.nan
    
    def _parse_pluplus_semicolon_list(self, s, outer_sep=";", inner_sep="++"):
        """Parse string into list of lists with ++ separator"""
        if pd.isna(s) or s == "":
            return []
        result = []
        for segment in str(s).split(outer_sep):
            if segment.strip() == "":
                result.append([])
            else:
                result.append(segment.split(inner_sep))
        return result
    
    def _parse_comma_semicolon_list(self, s, outer_sep=";", inner_sep=","):
        """Parse string into list of lists with comma separator"""
        if pd.isna(s) or s == "":
            return []
        result = []
        for segment in str(s).split(outer_sep):
            if segment.strip() == "":
                result.append([])
            else:
                result.append([x.strip() for x in segment.split(inner_sep)])
        return result


# ============================================
# PARSING PIPELINE COMPONENTS
# ============================================

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract static founder and company features.
    Now handles both lists and tuples.
    """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """No fitting needed; returns self."""
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transform the raw DataFrame X into the static feature set.
        """
        check_is_fitted(self, "_is_fitted")
        
        # Make a copy to avoid modifying the original data
        df = X.copy()

        # ------------------------------------------------------------------
        # Founder count features
        # ------------------------------------------------------------------
        df['num_founders'] = df['founders'].apply(self._count_items)

        df['num_female_founders'] = df['founders_genders'].apply(self._count_female)

        df['num_serial_founders'] = df['serial_founder'].apply(self._count_yes)

        df['num_founders_top_university'] = df['founder_top_university'].apply(self._count_yes)

        df['num_top_past_founders'] = df['top_past_founder'].apply(self._count_yes)

        df['num_repeated_founders'] = df['founders_first_company'].apply(self._count_no)

        df = df.rename(columns={"alumni_founders_10m": "num_alumni_founders_10m"})

        # ------------------------------------------------------------------
        # Universities
        # ------------------------------------------------------------------
        df['num_universities_total'] = df['founders_universities'].apply(self._count_total_items)
        df['num_universities_unique'] = df['founders_universities'].apply(self._count_unique_items)

        # ------------------------------------------------------------------
        # Education years (average, min, max)
        # ------------------------------------------------------------------
        edu_df = pd.DataFrame(df['founders_years_education'].apply(self._education_stats).tolist(), 
                             index=df.index,
                             columns=['founders_years_education_avg',
                                     'founders_years_education_min',
                                     'founders_years_education_max'])
        df = pd.concat([df, edu_df], axis=1)

        # ------------------------------------------------------------------
        # Years since first degree (using launch_year)
        # ------------------------------------------------------------------
        years_df = df.apply(self._years_since_degree, axis=1, result_type='expand')
        years_df.columns = ['founders_years_since_first_degree_avg',
                           'founders_years_since_first_degree_min',
                           'founders_years_since_first_degree_max']
        df = pd.concat([df, years_df], axis=1)

        # ------------------------------------------------------------------
        # Work experience (total positions, unique companies)
        # ------------------------------------------------------------------
        df['num_founders_experience_total'] = df['founders_company_experience'].apply(self._count_total_items)
        df['num_founders_experience_unique_companies'] = df['founders_company_experience'].apply(self._count_unique_items)

        # ------------------------------------------------------------------
        # Previous funding (exclude current company's funding)
        # ------------------------------------------------------------------
        funding_df = df.apply(self._prev_funding_stats, axis=1, result_type='expand')
        funding_df.columns = ['founders_prev_funding_avg',
                             'founders_prev_funding_min',
                             'founders_prev_funding_max']
        df = pd.concat([df, funding_df], axis=1)

        return df
    
    # Helper methods that work with both lists and tuples
    @staticmethod
    def _count_items(x):
        """Count items in a list or tuple"""
        if isinstance(x, (list, tuple)):
            return len(x)
        return 0

    @staticmethod
    def _count_yes(x):
        """Count 'yes' in a list or tuple"""
        if isinstance(x, (list, tuple)):
            return x.count('yes')
        return 0

    @staticmethod
    def _count_no(x):
        """Count 'no' in a list or tuple"""
        if isinstance(x, (list, tuple)):
            return x.count('no')
        return 0

    @staticmethod
    def _count_female(x):
        """Count female in a list or tuple"""
        if not isinstance(x, (list, tuple)):
            return 0
        return sum(1 for g in x if isinstance(g, str) and g.strip().lower() == 'female')

    @staticmethod
    def _count_total_items(x):
        """Count total items in a list of lists/tuples"""
        if not isinstance(x, (list, tuple)):
            return 0
        return sum(len(item) for item in x if isinstance(item, (list, tuple)))

    @staticmethod
    def _count_unique_items(x):
        """Count unique items in a list of lists/tuples"""
        if not isinstance(x, (list, tuple)):
            return 0
        all_items = []
        for item in x:
            if isinstance(item, (list, tuple)):
                all_items.extend(item)
        return len(set(all_items))

    @staticmethod
    def _education_stats(edu_list):
        """Calculate education stats from list or tuple"""
        if not isinstance(edu_list, (list, tuple)) or len(edu_list) == 0:
            return [np.nan, np.nan, np.nan]
        
        nums = [v for v in edu_list if pd.notna(v)]
        if not nums:
            return [np.nan, np.nan, np.nan]
        
        return [np.mean(nums), min(nums), max(nums)]

    def _years_since_degree(self, row):
        """Calculate years since first degree"""
        degree_years = row.get('founders_first_degree_year')
        launch_year = row.get('launch_year')
        
        if not isinstance(degree_years, (list, tuple)) or len(degree_years) == 0 or pd.isna(launch_year):
            return [np.nan, np.nan, np.nan]
        
        valid = [y for y in degree_years if pd.notna(y)]
        if len(valid) == 0:
            return [np.nan, np.nan, np.nan]
        
        diffs = [launch_year - y for y in valid]
        return [np.mean(diffs), min(diffs), max(diffs)]

    def _prev_funding_stats(self, row):
        """Calculate previous funding stats"""
        funding_list = row.get('founders_total_funding')
        current = row.get('total_funding_usd_m')
        
        if not isinstance(funding_list, (list, tuple)) or len(funding_list) == 0 or pd.isna(current):
            return [np.nan, np.nan, np.nan]
        
        prev_values = [max(v/1e6 - current, 0) for v in funding_list if pd.notna(v)]
        if len(prev_values) == 0:
            return [np.nan, np.nan, np.nan]
        
        return [np.mean(prev_values), min(prev_values), max(prev_values)]

class FXCacheMixin:
    def __init__(self, fx_cache_file="data/fx_rates.pkl",
                 base_currency="EUR", target_currency="USD"):
        self.fx_cache_file = fx_cache_file
        self.base_currency = base_currency.upper()
        self.target_currency = target_currency.upper()
    
    def get_fx_hash(self):
        """Create a hash of the FX file for cache invalidation"""
        import hashlib
        import os
        if os.path.exists(self.fx_cache_file):
            with open(self.fx_cache_file, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        return "no_file"

    def load_fx(self):
        """Load FX rates on-demand; returns a dict/Series"""
        try:
            fx_rates = pd.read_pickle(self.fx_cache_file)
        except FileNotFoundError:
            fx_rates = {}
        return fx_rates

    def convert_to_usd(self, amount, currency, date, fx_rates):
        if pd.isna(amount) or pd.isna(date):
            return np.nan
        if not currency or not isinstance(currency, str):
            return np.nan

        currency = currency.upper()

        # Ensure date is a Timestamp
        if not isinstance(date, pd.Timestamp):
            try:
                date = pd.to_datetime(date)
            except Exception:
                return np.nan

        if currency == "USD":
            return float(amount)

        # EUR → USD
        if currency == self.base_currency:
            usd_series = fx_rates.get(f"{self.base_currency}{self.target_currency}")
            if usd_series is None or usd_series.empty:
                return np.nan
            if date < usd_series.index.min() or date > usd_series.index.max():
                return np.nan
            rate = usd_series.asof(date)
            return float(amount * rate) if pd.notna(rate) else np.nan

        # Other currencies → EUR → USD
        series = fx_rates.get(f"{self.base_currency}{currency}")
        if series is None or series.empty:
            return np.nan
        if date < series.index.min() or date > series.index.max():
            return np.nan
        rate_to_eur = series.asof(date)
        if pd.isna(rate_to_eur) or rate_to_eur == 0:
            return np.nan

        eur_amount = amount / rate_to_eur

        usd_series = fx_rates.get(f"{self.base_currency}{self.target_currency}")
        if usd_series is None or usd_series.empty:
            return np.nan
        if date < usd_series.index.min() or date > usd_series.index.max():
            return np.nan
        usd_rate = usd_series.asof(date)
        if pd.isna(usd_rate):
            return np.nan

        return float(eur_amount * usd_rate)
    
class BuyInRoundExtractor(BaseEstimator, TransformerMixin, FXCacheMixin):
    def __init__(self, fx_cache_file="data/fx_rates.pkl", buyin_round_number=1):
        FXCacheMixin.__init__(self, fx_cache_file=fx_cache_file)
        self.buyin_round_number = buyin_round_number

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        
        # Merge the buyin round to the original data
        expanded_rounds = self.expand_rounds(X)
        buyin_rounds = expanded_rounds[expanded_rounds["buyin_flag"] == 1]
        merged = X.merge(buyin_rounds, on=["id"], how="left").drop(columns=['buyin_flag'])
        
        # Rename columns
        rename_dict = {
            "round_number": "buyin_round_number",
            "round_date": "buyin_date",
            "round_amount_original": "buyin_amount_original",
            "round_amount_usd": "buyin_amount_usd",
            "round_currency": "buyin_currency",
            "round_type": "buyin_type"
        }
        
        # Rename the columns
        renamed = merged.rename(columns=rename_dict)
        
        # --- SORT FINAL DATAFRAME FOR TIME-AWARE SPLITTING ---
        renamed = renamed.sort_values("buyin_date").reset_index(drop=True)

        return renamed

    def expand_rounds(self, X):
        fx_rates = self.load_fx()
        
        all_rows = []

        for _, row in X.iterrows():
            id_ = row.get("id")

            # Build round-level records
            rounds = []
            for rd, ra, rc, rt, ri in zip_longest(
                row.get("round_dates", []),
                row.get("round_amounts", []),
                row.get("round_currencies", []),
                row.get("round_types", []),
                row.get("round_investors", [])
            ):
                if pd.notna(rd):
                    rd_ts = pd.to_datetime(rd)

                    usd_amount = np.nan
                    usd_amount = self.convert_to_usd(
                        amount=ra,
                        currency=rc,
                        date=rd_ts,
                        fx_rates=fx_rates
                    )

                    rounds.append({
                        "id": id_,
                        "round_date": rd_ts,
                        "amount_original": ra,
                        "amount_usd": usd_amount,
                        "currency": rc,
                        "type": rt,
                        "investors": ri
                    })

            if not rounds:
                continue

            # Sort by date
            rounds.sort(key=lambda r: r["round_date"])

            # Compute past valuations and funding features
            valuations = [
                (pd.to_datetime(d), v)
                for d, v in zip(row.get("valuation_dates", []), row.get("valuation_values_usd", []))
                if pd.notna(d) and pd.notna(v)
            ]
            valuations.sort(key=lambda x: x[0])
            max_valuation_so_far = -np.inf

            for round_number, r in enumerate(rounds, start=1):
                round_date = r["round_date"]

                # Valuations before this round
                vals_before = [(d, v) for d, v in valuations if d < round_date]
                last_valuation_usd = vals_before[-1][1] if vals_before else np.nan
                max_valuation_so_far = max([v for _, v in vals_before], default=np.nan)

                # Time since last valuation
                if vals_before:
                    last_val_date = vals_before[-1][0]
                    months_since_last_valuation = (
                        (round_date.year - last_val_date.year) * 12 +
                        (round_date.month - last_val_date.month)
                    )
                    if round_date.day < last_val_date.day:
                        months_since_last_valuation -= 1
                else:
                    months_since_last_valuation = np.nan

                # Time since last funding
                past_rounds = [r_ for r_ in rounds if r_["round_date"] < round_date]
                months_since_last_funding = np.nan
                if past_rounds:
                    prev_date = past_rounds[-1]["round_date"]
                    months_since_last_funding = (
                        (round_date.year - prev_date.year) * 12 +
                        (round_date.month - prev_date.month)
                    )
                    if round_date.day < prev_date.day:
                        months_since_last_funding -= 1
                past_amounts = [r_["amount_usd"] for r_ in past_rounds]
                past_investors = [inv for r_ in past_rounds for inv in r_["investors"]]

                # Unique investors
                unique_investors = set(past_investors)
                independent_investments = len(past_investors)

                all_rows.append({
                    "id": id_,
                    "round_number": round_number,
                    "round_date": round_date,
                    "round_amount_original": r["amount_original"],
                    "round_amount_usd": r["amount_usd"],
                    "round_currency": r["currency"],
                    "round_type": r["type"],
                    "past_months_since_last_valuation": months_since_last_valuation,
                    "past_last_valuation_usd": last_valuation_usd,
                    "past_highest_valuation_usd": max_valuation_so_far,
                    "past_months_since_last_funding": months_since_last_funding,
                    "past_total_funding_usd": np.nansum(past_amounts),
                    "past_num_unique_investors": len(unique_investors),
                    "past_num_independent_investments": independent_investments
                })

        # Build dataframe
        df_expanded = pd.DataFrame(all_rows)
        if df_expanded.empty:
            return df_expanded

        df_expanded.sort_values(["id", "round_number"], inplace=True)
        df_expanded.reset_index(drop=True, inplace=True)

        # Initialize flag column
        df_expanded["buyin_flag"] = 0

        # Flag rows where round_number matches
        df_expanded.loc[df_expanded["round_number"] == self.buyin_round_number, "buyin_flag"] = 1

        return df_expanded

class TargetExtractor(FXCacheMixin, BaseEstimator, TransformerMixin):
    """
    Expands company data to event-level and computes forward-looking regression
    targets per buy-in round using FX-converted amounts from cached FX rates.
 
    Censors rows where the observation window extends beyond the download date.
    Does NOT compute binary quantile flags — use QuantileTargetBinarizer after
    splitting to avoid data leakage.
    """
 
    def __init__(
        self,
        fx_cache_file="data/fx_rates.pkl",
        window_years=5,
        download_date=None,
        exit_round_types=None,
    ):
        FXCacheMixin.__init__(self, fx_cache_file=fx_cache_file)
        self.window_years = window_years
        self.download_date = pd.to_datetime(download_date)
        self.exit_round_types = exit_round_types
 
    def fit(self, X, y=None):
        return self  # Stateless — no fitting required
 
    def transform(self, X):
        # 1. Expand events and build regression targets
        events = self.expand_events(X)
        targets = self.build_targets(events).drop(columns=["buyin_date"])
 
        # 2. Merge targets back to original X
        X_out = X.merge(targets, on="id", how="left")
 
        # 3. Censor rows where the observation window exceeds the download date
        censored_mask = (
            X_out["buyin_date"] + pd.DateOffset(years=self.window_years)
            > self.download_date
        )
        print(
            f"   - censored rows removed: {censored_mask.sum()} out of {len(X_out)}"
        )
        return X_out[~censored_mask]
        
    def expand_events(self, X):
        fx_rates = self.load_fx()
        
        all_rows = []

        for _, row in X.iterrows():
            cid = row["id"]
            buyin_date = row["buyin_date"]
            num_founders = row["num_founders"]
            
            # ---- Valuations
            for d, v in zip(row.get("valuation_dates", []), row.get("valuation_values_usd", [])):
                if pd.notna(d) and pd.notna(v):
                    all_rows.append({
                        "id": cid,
                        "buyin_date": buyin_date,
                        "num_founders": num_founders,
                        "event_type": "valuation",
                        "date": pd.to_datetime(d),
                        "valuation_usd": v
                    })

            # ---- Rounds
            for rd, ra, rc, rt in zip_longest(
                row.get("round_dates", []),
                row.get("round_amounts", []),
                row.get("round_currencies", []),
                row.get("round_types", [])
            ):
                if pd.notna(rd):
                    rd_ts = pd.to_datetime(rd)
                    usd_amount = self.convert_to_usd(ra, rc, rd_ts, fx_rates)
                    all_rows.append({
                        "id": cid,
                        "buyin_date": buyin_date,
                        "num_founders": num_founders,
                        "event_type": "round",
                        "date": rd_ts,
                        "amount_usd": usd_amount,
                        "round_type": rt
                    })

            # ---- Employees
            for d, e in zip(row.get("employee_dates", []), row.get("employee_values", [])):
                if pd.notna(d) and pd.notna(e):
                    all_rows.append({
                        "id": cid,
                        "buyin_date": buyin_date,
                        "num_founders": num_founders,
                        "event_type": "employee",
                        "date": pd.to_datetime(d),
                        "employees": e
                    })

        events = pd.DataFrame(all_rows)
        if events.empty:
            return events

        events.sort_values(["id", "date"], inplace=True)     
        events["cumulative_funding_usd"] = events.groupby("id")["amount_usd"].cumsum()

        return events

    def build_targets(self, events):
    # If events is empty, return a DataFrame with only the id column (will be merged later)
        if events.empty:
            continuous_target_cols = [
                "target_valuation_delta",
                "target_funding_delta",
                "target_employee_delta"
            ]
            return pd.DataFrame(columns=["id", "window_end"] + continuous_target_cols + ["target_has_exit", "target_has_round"])
        results = []
        for cid, group in events.groupby("id"):
            group = group.sort_values("date").reset_index(drop=True)

            buyin_date = group["buyin_date"].iloc[0] if "buyin_date" in group.columns else pd.NaT
            window_end = buyin_date + pd.DateOffset(years=self.window_years) if pd.notna(buyin_date) else pd.NaT

            if pd.notna(buyin_date):
                pre_window = group[group["date"] <= buyin_date]
                window = group[(group["date"] > buyin_date) & (group["date"] <= window_end)]
                post_window = group[group["date"] > window_end] if pd.notna(window_end) else pd.DataFrame()
            else:
                pre_window = pd.DataFrame()
                window = pd.DataFrame()
                post_window = pd.DataFrame()
            
            # ----- Valuation -----
            # Last valuation in window (value and date)
            last_val_window = np.nan
            last_val_window_date = pd.NaT
            if not window.empty and "event_type" in window.columns and "valuation_usd" in window.columns:
                val_window_rows = window[window["event_type"] == "valuation"]
                if not val_window_rows.empty:
                    last_val_window = val_window_rows["valuation_usd"].iloc[-1]
                    last_val_window_date = val_window_rows["date"].iloc[-1]

            # Last valuation before window (value and date)
            last_val_pre = np.nan
            last_val_pre_date = pd.NaT
            if not pre_window.empty and "event_type" in pre_window.columns and "valuation_usd" in pre_window.columns:
                val_pre_rows = pre_window[pre_window["event_type"] == "valuation"]
                if not val_pre_rows.empty:
                    last_val_pre = val_pre_rows["valuation_usd"].iloc[-1]
                    last_val_pre_date = val_pre_rows["date"].iloc[-1]

            # First valuation after window (value and date)
            first_val_post = np.nan
            first_val_post_date = pd.NaT
            if not post_window.empty and "event_type" in post_window.columns and "valuation_usd" in post_window.columns:
                val_post_rows = post_window[post_window["event_type"] == "valuation"]
                if not val_post_rows.empty:
                    first_val_post = val_post_rows["valuation_usd"].iloc[0]
                    first_val_post_date = val_post_rows["date"].iloc[0]

            # Determine the best valuation before/at window_end and baseline
            if pd.notna(last_val_window):
                if pd.notna(last_val_pre):
                    baseline_val = last_val_pre
                else:
                    baseline_val = 0
                best_before_val = last_val_window
                best_before_val_date = last_val_window_date
            elif pd.notna(last_val_pre):
                baseline_val = last_val_pre
                best_before_val = last_val_pre
                best_before_val_date = last_val_pre_date
            else:
                baseline_val = 0
                best_before_val = 0
                best_before_val_date = buyin_date

            # Try extrapolation if we have both before and after points
            estimated_val_end = np.nan
            if pd.notna(first_val_post):
                # Linear interpolation between best_before and first_post
                t0 = best_before_val_date.timestamp()
                t1 = first_val_post_date.timestamp()
                t_target = window_end.timestamp()
                if t1 != t0:
                    ratio = (t_target - t0) / (t1 - t0)
                    estimated_val_end = best_before_val + (first_val_post - best_before_val) * ratio
                else:
                    estimated_val_end = best_before_val

            # Compute delta_val
            if pd.notna(estimated_val_end):
                delta_val = estimated_val_end - baseline_val
            elif pd.notna(last_val_window):
                delta_val = last_val_window - baseline_val
            else:
                delta_val = np.nan        
              
            # ----- Funding rounds -----
            # (seperate 0 classification and non zero regression bc zero-inflated)
            if not window.empty and "event_type" in window.columns:
                funding_round_rows = window[window["event_type"] == "round"]
                has_rounds = not funding_round_rows.empty
            else:
                funding_round_rows = pd.DataFrame()
                has_rounds = False

            if has_rounds:
                if "amount_usd" in funding_round_rows.columns:
                    positive_amounts = funding_round_rows.loc[funding_round_rows["amount_usd"] > 0, "amount_usd"]
                    if not positive_amounts.empty:
                        delta_fund = positive_amounts.sum()
                    else:
                        delta_fund = np.nan
                else:
                    delta_fund = np.nan
            else:
                delta_fund = np.nan

            # ----- Employees (with interpolation) -----
            # Last employee in window (value and date)
            last_emp_window = np.nan
            last_emp_window_date = pd.NaT
            if not window.empty and "event_type" in window.columns and "employees" in window.columns:
                emp_window_rows = window[window["event_type"] == "employee"]
                if not emp_window_rows.empty:
                    last_emp_window = emp_window_rows["employees"].iloc[-1]
                    last_emp_window_date = emp_window_rows["date"].iloc[-1]

            # Last employee before window (value and date)
            last_emp_pre = np.nan
            last_emp_pre_date = pd.NaT
            if not pre_window.empty and "event_type" in pre_window.columns and "employees" in pre_window.columns:
                emp_pre_rows = pre_window[pre_window["event_type"] == "employee"]
                if not emp_pre_rows.empty:
                    last_emp_pre = emp_pre_rows["employees"].iloc[-1]
                    last_emp_pre_date = emp_pre_rows["date"].iloc[-1]

            # First employee after window (value and date)
            first_emp_post = np.nan
            first_emp_post_date = pd.NaT
            if not post_window.empty and "event_type" in post_window.columns and "employees" in post_window.columns:
                emp_post_rows = post_window[post_window["event_type"] == "employee"]
                if not emp_post_rows.empty:
                    first_emp_post = emp_post_rows["employees"].iloc[0]
                    first_emp_post_date = emp_post_rows["date"].iloc[0]

            # Determine the best employee count before/at window_end
            if pd.notna(last_emp_window): # if there is an employee value in window
                if pd.notna(last_emp_pre):
                    baseline_emp = last_emp_pre
                else:
                    baseline_emp = group["num_founders"].iloc[0] if "num_founders" in group.columns else 0
                best_before_emp = last_emp_window
                best_before_emp_date = last_emp_window_date
            elif pd.notna(last_emp_pre): # if there is an employee value before the window
                baseline_emp = last_emp_pre
                best_before_emp = last_emp_pre
                best_before_emp_date = last_emp_pre_date
            else: # take the num_founders baseline at buyin
                baseline_emp = group["num_founders"].iloc[0] if "num_founders" in group.columns else 0
                best_before_emp = group["num_founders"].iloc[0] if "num_founders" in group.columns else 0
                best_before_emp_date = buyin_date

            # Try extrapolation if we have both before and after points
            estimated_emp_end = np.nan
            if pd.notna(first_emp_post):
                # Linear interpolation between best_before and first_post
                t0 = best_before_emp_date.timestamp()
                t1 = first_emp_post_date.timestamp()
                t_target = window_end.timestamp()
                if t1 != t0:
                    ratio = (t_target - t0) / (t1 - t0)
                    estimated_emp_end = best_before_emp + (first_emp_post - best_before_emp) * ratio
                else:
                    estimated_emp_end = best_before_emp

            # Compute delta_emp
            if pd.notna(estimated_emp_end):
                delta_emp = estimated_emp_end - baseline_emp
            elif pd.notna(last_emp_window):
                delta_emp = last_emp_window - baseline_emp
            else:
                delta_emp = np.nan
            
            # Determine success flag (exit event) safely
            if not window.empty and "event_type" in window.columns and "round_type" in window.columns:
                exit_round_rows = window[window["event_type"] == "round"]
                if not exit_round_rows.empty:
                    has_bankruptcy = exit_round_rows["round_type"].isin(["BANKRUPTCY"]).any()
                    has_exit = (
                        exit_round_rows["round_type"].isin(self.exit_round_types).any()
                        and not has_bankruptcy
                    )
                else:
                    has_exit = False
            else:
                has_exit = False

            results.append({
                "id": cid,
                "buyin_date": buyin_date,  # for debugging, drop later because already in data
                "window_end": window_end,
                "target_has_exit": has_exit,
                
                "target_has_round": has_rounds,
                "target_funding_delta": delta_fund,
                
                "target_valuation_delta": delta_val,
                
                "target_employee_delta": delta_emp,
            })

        df = pd.DataFrame(results)
        for col in ["target_has_exit", "target_has_round"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
                
        return df

# ============================================
# BINARIZER PIPELINE COMPONENTS
# ============================================

class QuantileTargetBinarizer(BaseEstimator, TransformerMixin):
    """
    Fits quantile thresholds on training data and creates binary flag columns
    for the specified (or auto-detected) continuous target columns.
 
    Fit after train/test splitting to avoid data leakage. The thresholds are
    computed exclusively on the training fold and then applied consistently
    to any split.
 
    Parameters
    ----------
    binary_targets : list[str] or None
        Explicit list of target columns to binarise. If None, all columns
        whose names start with 'target_' are used, excluding 'target_has_exit'
        and 'target_has_round'.
    quantile_threshold : float
        Quantile cutoff for the upper-quantile flag, e.g. 0.75 for top 25%.
    """
 
    def __init__(self, binary_targets=None, quantile_threshold=0.75):
        self.binary_targets = binary_targets
        self.quantile_threshold = quantile_threshold
 
    def fit(self, X, y=None):
        # Resolve which columns to binarise
        if self.binary_targets is None:
            exclude = {"target_has_exit", "target_has_round"}
            self.binary_targets_ = [
                col
                for col in X.columns
                if col.startswith("target_") and col not in exclude
            ]
        else:
            self.binary_targets_ = list(self.binary_targets)
 
        # Compute quantile thresholds on the training fold only
        self.thresholds_ = {
            col: X[col].quantile(self.quantile_threshold, interpolation="higher")
            for col in self.binary_targets_
        }
 
        print(f"   - quantile thresholds fitted: {self.thresholds_}")
        return self
 
    def transform(self, X):
        check_is_fitted(self, "thresholds_")
        X_out = X.copy()
 
        for col in self.binary_targets_:
            bin_col = f"{col}_upper_q"
            X_out[bin_col] = X_out[col].apply(
                lambda x: (
                    1 if pd.notna(x) and x >= self.thresholds_[col]
                    else (0 if pd.notna(x) else np.nan)
                )
            )
 
        return X_out

# ============================================
# TUNABLE PIPELINE COMPONENTS
# ============================================

class TenYearTBYMerger(BaseEstimator, TransformerMixin):
    """
    Merge selected 10Y TBY series from CSV into main DataFrame by buyin year,
    optionally renaming columns according to a mapping.
    """

    def __init__(self,
                 file_path,
                 buyin_date_column="buyin_date",
                 macro_year_column="Period",
                 macro_value_column="Average",
                 columns_to_merge=None,  # dict {Series in CSV: new column name}
                 merge=True):
        self.file_path = file_path
        self.buyin_date_column = buyin_date_column
        self.macro_year_column = macro_year_column
        self.macro_value_column = macro_value_column
        self.columns_to_merge = columns_to_merge
        self.merge = merge

    def fit(self, X, y=None):
        # Load CSV
        df_macro = pd.read_csv(self.file_path, decimal=",")

        if self.columns_to_merge:
            # Filter CSV to only the requested Series
            df_macro = df_macro[df_macro['Series'].isin(self.columns_to_merge.keys())]

        # Pivot so each Series becomes a column
        self.df_macro_ = df_macro.pivot(
            index=self.macro_year_column,
            columns='Series',
            values=self.macro_value_column
        ).reset_index()

        # Rename columns according to mapping
        if self.columns_to_merge:
            rename_mapping = {k: v for k, v in self.columns_to_merge.items() if k in self.df_macro_.columns}
            self.df_macro_.rename(columns=rename_mapping, inplace=True)

        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        
        if not self.merge:
            return X.copy()

        X = X.copy()
        # Convert buyin_date to datetime and extract year
        X[self.buyin_date_column] = pd.to_datetime(X[self.buyin_date_column])
        X['buyin_year'] = X[self.buyin_date_column].dt.year

        # Merge pivoted macro table on buyin_year
        df_macro = self.df_macro_.copy()

        # Merge
        X = X.merge(df_macro, left_on='buyin_year', right_on=self.macro_year_column, how='left')

        # Drop the macro year column
        X.drop(columns=['buyin_year', self.macro_year_column], inplace=True, errors='ignore')
        return X

class EmbeddingScoreMerger(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    input_type : {'score', 'emb', 'emb_score'}
    embedding_paths : dict {model_name: path} | None
    embedding_model : str | None
        Name of a single model (must be a key in embedding_paths),
    scoring_paths : dict {model_name: path} | None
    scoring_model : str | None
        Name of a single model (must be a key in scoring_paths),
        or "avg" to average all models. None defaults to "avg".
    embedding_pca_dim : int | None
    """

    def __init__(
        self,
        input_type="score",
        embedding_paths=None,
        embedding_model=None,       # must be an explicit key in embedding_paths
        scoring_paths=None,
        scoring_model=None,         # key in scoring_paths, or "avg" / None
        embedding_pca_dim=None,
    ):
        self.input_type = input_type
        self.embedding_paths = embedding_paths
        self.embedding_model = embedding_model
        self.scoring_paths = scoring_paths
        self.scoring_model = scoring_model
        self.embedding_pca_dim = embedding_pca_dim

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        self.processed_scores_ = None
        self.embeddings_ = None
        self.pca_ = None

        needs_scores = self.input_type in ("score", "emb_score")
        needs_emb    = self.input_type in ("emb",   "emb_score")

        # ── scores ────────────────────────────────────────────────────
        if needs_scores:
            if not self.scoring_paths:
                raise ValueError("scoring_paths must be provided for input_type='score'/'emb_score'")

            model = self.scoring_model or "avg"
            if model != "avg" and model not in self.scoring_paths:
                raise ValueError(
                    f"scoring_model '{model}' not found in scoring_paths keys: {list(self.scoring_paths)}"
                )

            self.processed_scores_ = self._load_scores(self.scoring_paths, model)

        # ── embeddings ────────────────────────────────────────────────
        if needs_emb:
            if not self.embedding_paths:
                raise ValueError("embedding_paths must be provided for input_type='emb'/'emb_score'")

            if self.embedding_model is None:
                raise ValueError(
                    f"embedding_model must be specified. Available keys: {list(self.embedding_paths)}"
                )
            if self.embedding_model not in self.embedding_paths:
                raise ValueError(
                    f"embedding_model '{self.embedding_model}' not found in embedding_paths keys: {list(self.embedding_paths)}"
                )

            selected_path = self.embedding_paths[self.embedding_model]

            if selected_path not in _EMBEDDING_CACHE:
                _EMBEDDING_CACHE[selected_path] = pd.read_pickle(selected_path)

            cache_key = (selected_path, self.embedding_pca_dim)
            if cache_key not in _PCA_CACHE:
                emb_df   = _EMBEDDING_CACHE[selected_path]
                emb_cols = [c for c in emb_df.columns if c != "id"]
                nan_mask      = emb_df[emb_cols].isna().any(axis=1)
                complete_rows = emb_df[~nan_mask]

                if len(complete_rows) == 0:
                    raise ValueError("All embedding rows contain NaNs – cannot fit PCA")

                emb_matrix = complete_rows[emb_cols].values

                if self.embedding_pca_dim:
                    pca = PCA(n_components=self.embedding_pca_dim, random_state=42)
                    reduced = pca.fit_transform(emb_matrix)
                    prefix   = self.embedding_model
                    pca_cols = [
                        f"{prefix}_embedding_pca_{i}_of_{self.embedding_pca_dim}"
                        for i in range(self.embedding_pca_dim)
                    ]
                    out_df = pd.DataFrame(index=emb_df.index, columns=pca_cols, dtype=float)
                    out_df.loc[~nan_mask, pca_cols] = reduced
                    out_df["id"] = emb_df["id"].values
                else:
                    pca    = None
                    out_df = emb_df.copy()

                _PCA_CACHE[cache_key] = (pca, out_df)

            self.pca_, self.embeddings_ = _PCA_CACHE[cache_key]

        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, X, id_column="id"):
        result = X.copy()

        if self.input_type in ("score", "emb_score"):
            check_is_fitted(self, "processed_scores_")
            result = result.merge(self.processed_scores_, on=id_column, how="left")

        if self.input_type in ("emb", "emb_score"):
            check_is_fitted(self, "embeddings_")
            result = result.merge(self.embeddings_, on=id_column, how="left")

        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_scores(scoring_paths, model):
        """
        model == "avg"  → load all files, outer-join on id, return avg_{score} columns.
        model == <name> → load that file only, return {model}_{score} columns.
        Missing ids in any file become NaN.
        """
        def _read(path):
            if path not in _SCORE_CACHE:
                raw = pd.read_pickle(path)
                if isinstance(raw, dict):
                    raw = pd.DataFrame(raw['results'])
                _SCORE_CACHE[path] = raw
            return _SCORE_CACHE[path]

        def _score_cols(df):
            cols = [c for c in df.columns if "score" in c.lower()]
            if not cols:
                raise ValueError(f"No columns containing 'score' found.")
            return cols

        if model != "avg":
            raw        = _read(scoring_paths[model])
            score_cols = _score_cols(raw)
            prefix     = model.lower().replace("-", "_")
            return (
                raw[["id"] + score_cols]
                .rename(columns={c: f"{prefix}_{c}" for c in score_cols})
            )

        # "avg": load every model, outer-join, average matching score columns
        per_model = {}
        for name, path in scoring_paths.items():
            raw        = _read(path)
            score_cols = _score_cols(raw)
            prefix     = name.lower().replace("-", "_")
            per_model[name] = (
                raw[["id"] + score_cols]
                .rename(columns={c: f"{prefix}_{c}" for c in score_cols})
            )

        merged = None
        for df in per_model.values():
            merged = df if merged is None else merged.merge(df, on="id", how="outer")

        # Derive base score names from the first model and average across all
        first_prefix = next(iter(per_model)).lower().replace("-", "_")
        base_scores  = [
            col[len(first_prefix) + 1:]          # strip "prefix_"
            for col in next(iter(per_model.values())).columns
            if col != "id"
        ]

        result = pd.DataFrame({"id": merged["id"]})
        for base in base_scores:
            cols = [
                f"{n.lower().replace('-', '_')}_{base}"
                for n in per_model
                if f"{n.lower().replace('-', '_')}_{base}" in merged.columns
            ]
            result[f"avg_{base}"] = merged[cols].mean(axis=1)

        return result

class CatEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        cat_columns=None,
        cat_list_columns=None,
        count_list_columns=None,
        count_lol_columns=None,
        remove_cat=None,
        remove_string=None,
        remove_cat_with_string=None,
        min_frequency=1
    ):

        self.cat_columns = cat_columns
        self.cat_list_columns = cat_list_columns
        self.count_list_columns = count_list_columns
        self.count_lol_columns = count_lol_columns

        self.remove_cat = remove_cat
        self.remove_string = remove_string
        self.remove_cat_with_string = remove_cat_with_string

        self.min_frequency = min_frequency

        self.scalar_encoders = {}
        self.cat_list_encoders = {}
        self.count_list_encoders = {}
        self.count_lol_encoders = {}

        self.feature_names_ = None
        
        self.infrequent_counts_ = {}
        
        self.na_values = {"n/a", "na", "nan", "none", "unknown", "-", ""}


    # ------------------------------------------------
    # FIT
    # ------------------------------------------------
    def fit(self, X, y=None):

        X = pd.DataFrame(X).copy()
        X = self._clean_inputs(X)

        # --- scalar categorical ---
        # OneHotEncoder's min_frequency collapses rare categories into
        # "infrequent_sklearn" automatically when min_frequency > 1.
        for col in self.cat_columns:
            ohe = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
                min_frequency=self.min_frequency,
            )
            col_data = X[[col]].copy()
            col_data[col] = col_data[col].where(
                col_data[col].str.lower() != 'nan', other=None
            )
            ohe.fit(col_data.dropna())
            self.scalar_encoders[col] = ohe
            n_infreq = len(ohe.infrequent_categories_[0]) if ohe.infrequent_categories_[0] is not None else 0
            self.infrequent_counts_[col] = n_infreq
            
        # --- list multi-hot ---
        for col in self.cat_list_columns:

            counter = Counter()
            for row in X[col]:
                counter.update(row)

            classes = sorted([k for k, v in counter.items() if v >= self.min_frequency])
            n_total = len(counter)
            n_kept = len(classes)
            n_infreq = n_total - n_kept

            self.cat_list_encoders[col] = classes
            self.infrequent_counts_[col] = n_infreq

        # --- list count ---
        for col in self.count_list_columns:

            counter = Counter()
            for row in X[col]:
                counter.update(row)

            classes = sorted([k for k, v in counter.items() if v >= self.min_frequency])
            n_total = len(counter)
            n_kept = len(classes)
            n_infreq = n_total - n_kept

            self.count_list_encoders[col] = classes
            self.infrequent_counts_[col] = n_infreq

        # --- list-of-lists count ---
        for col in self.count_lol_columns:

            flattened = X[col].apply(self._flatten_list_of_lists)
            counter = Counter()
            for row in flattened:
                counter.update(row)

            classes = sorted([k for k, v in counter.items() if v >= self.min_frequency])
            n_total = len(counter)
            n_kept = len(classes)
            n_infreq = n_total - n_kept

            self.count_lol_encoders[col] = classes
            self.infrequent_counts_[col] = n_infreq

        self._is_fitted = True

        # --- build feature names ---
        self.feature_names_ = self._build_feature_names()

        return self


    # ------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------
    def transform(self, X):

        check_is_fitted(self, "_is_fitted")

        X = pd.DataFrame(X).copy()
        X = self._clean_inputs(X)

        arrays = []
        feature_names = []

        # --- scalar categorical ---
        for col, enc in self.scalar_encoders.items():
            col_data = X[[col]].copy()
            col_data[col] = col_data[col].where(
                col_data[col].str.lower() != 'nan', other=None
            )
            nan_mask = col_data[col].isna().values
            col_data[col] = col_data[col].fillna(enc.categories_[0][0])  # temp fill
            arr = enc.transform(col_data)
            arr[nan_mask] = np.nan
            arrays.append(arr)
            feature_names.extend(list(enc.get_feature_names_out([col])))

        # --- list multi-hot ---
        # Empty lists are treated as missing: all columns set to NaN,
        # consistent with how scalar categoricals handle NaN inputs.
        for col, classes in self.cat_list_encoders.items():
            arr = []
            for row in X[col]:
                if not row:  # empty list → all NaN
                    arr.append([np.nan] * len(classes))
                else:
                    s = set(row)
                    arr.append([1 if cls in s else 0 for cls in classes])
            arrays.append(np.array(arr))
            feature_names.extend([f"{col}_{cls}" for cls in classes])

        # --- list count ---
        for col, classes in self.count_list_encoders.items():

            known = set(classes)
            arr = []

            for row in X[col]:
                if not row:  # empty list → all NaN
                    arr.append([np.nan] * (len(classes) + 1))  # +1 for infrequent_count
                else:
                    counts = Counter(row)
                    vec = [counts.get(cls, 0) for cls in classes]
                    other_count = sum(v for k, v in counts.items() if k not in known)
                    vec.append(other_count)
                    arr.append(vec)

            arrays.append(np.array(arr))
            feature_names.extend([f"{col}_{cls}_count" for cls in classes])
            feature_names.append(f"{col}_infrequent_count")

        # --- list-of-lists count ---
        for col, classes in self.count_lol_encoders.items():

            known = set(classes)
            flattened = X[col].apply(self._flatten_list_of_lists)
            arr = []

            for row in flattened:
                if not row:  # empty list → all NaN
                    arr.append([np.nan] * (len(classes) + 1))  # +1 for infrequent_count
                else:
                    counts = Counter(row)
                    vec = [counts.get(cls, 0) for cls in classes]
                    other_count = sum(v for k, v in counts.items() if k not in known)
                    vec.append(other_count)
                    arr.append(vec)

            arrays.append(np.array(arr))
            feature_names.extend([f"{col}_{cls}_count" for cls in classes])
            feature_names.append(f"{col}_infrequent_count")

        self.feature_names_ = feature_names

        if arrays:
            return pd.DataFrame(
                np.hstack(arrays),
                columns=feature_names,
                index=X.index
            )

        return pd.DataFrame(index=X.index)


    # ------------------------------------------------
    # FEATURE NAMES
    # ------------------------------------------------

    def get_feature_names_out(self, input_features=None):

        if self.feature_names_ is None:
            raise AttributeError("Call transform first.")

        return np.array(self.feature_names_)


    # ------------------------------------------------
    # HELPERS
    # ------------------------------------------------

    def _build_feature_names(self):
        """Build feature name list after fit, mirroring transform order."""
        names = []

        for col, enc in self.scalar_encoders.items():
            # get_feature_names_out already returns prefixed names like "col_value"
            names.extend(list(enc.get_feature_names_out([col])))

        for col, classes in self.cat_list_encoders.items():
            names.extend([f"{col}_{cls}" for cls in classes])
            # Dont encode missingness to avoid leakage

        for col, classes in self.count_list_encoders.items():
            names.extend([f"{col}_{cls}_count" for cls in classes])
            names.append(f"{col}_infrequent_count")

        for col, classes in self.count_lol_encoders.items():
            names.extend([f"{col}_{cls}_count" for cls in classes])
            names.append(f"{col}_infrequent_count")

        return names


    # ------------------------------------------------
    # CLEANING
    # ------------------------------------------------

    def _clean_inputs(self, X):

        for col in self.cat_columns:
            X[col] = X[col].apply(self._clean_scalar)

        for col in self.cat_list_columns + self.count_list_columns:
            X[col] = X[col].apply(self._clean_list)

        for col in self.count_lol_columns:
            X[col] = X[col].apply(self._clean_lol)

        return X


    def _clean_scalar(self, x):
        if pd.isna(x):
            return x

        val = str(x)
        for s in self.remove_string:
            val = val.replace(s, "")
        val = val.strip()

        if val.lower() in self.na_values:
            return None

        if any(s in val for s in self.remove_cat_with_string):
            return None
        if val in self.remove_cat or val == "":
            return None

        return val


    def _clean_list(self, lst):
        if not isinstance(lst, list):
            return []
        cleaned = []
        for item in lst:
            val = str(item)
            for s in self.remove_string:
                val = val.replace(s, "")
            val = val.strip()

            if val.lower() in self.na_values:
                continue

            if any(s in val for s in self.remove_cat_with_string):
                continue
            if val in self.remove_cat or val == "":
                continue
            cleaned.append(val)
        return cleaned


    def _clean_lol(self, lst):
        if not isinstance(lst, list):
            return []

        cleaned = []

        for sub in lst:
            if isinstance(sub, list):
                sub_clean = []
                for item in sub:
                    val = str(item)
                    for s in self.remove_string:
                        val = val.replace(s, "")
                    val = val.strip()

                    if val.lower() in self.na_values:
                        continue
                    if any(s in val for s in self.remove_cat_with_string):
                        continue
                    if val in self.remove_cat or val == "":
                        continue

                    sub_clean.append(val)

                if sub_clean:
                    cleaned.append(sub_clean)

            else:
                val = str(sub)
                for s in self.remove_string:
                    val = val.replace(s, "")
                val = val.strip()

                if val.lower() in self.na_values:
                    continue
                if any(s in val for s in self.remove_cat_with_string):
                    continue
                if val not in self.remove_cat and val != "":
                    cleaned.append([val])

        return cleaned


    def _flatten_list_of_lists(self, x):

        if not isinstance(x, list):
            return []

        flat = []

        for sub in x:

            if isinstance(sub, list):
                flat.extend(sub)
            else:
                flat.append(sub)

        return flat

class TargetColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drop any column whose name contains the substring 'target'.
    """
    def __init__(self, substring='target'):
        self.substring = substring

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_is_fitted")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        cols_to_drop = [col for col in X.columns if self.substring in col]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        
        return X

class EarlyStoppingCallback(TrainingCallback):
    """
    Stop training when no improvement exceeds min_abs_improvement for `rounds`
    consecutive rounds — patience-based, handles both minimize and
    maximize metrics correctly.

    Parameters
    ----------
    rounds : int
        Number of rounds without improvement before stopping.
    min_abs_improvement : float
        Minimum absolute improvement to count as a new best.
    metric_name : str | None
        Metric to watch. None = last metric in eval results.
    maximize : bool
        True if higher is better (auc), False if lower (logloss, rmse).
    smooth_window : int
        Rolling average window applied before comparing scores.
    warmup : int
        Warmup period — early stopping cannot fire before this many rounds.
        Prevents a half-empty smoothing window from locking in an early
        spike as best_iteration.
    """

    def __init__(
        self,
        rounds: int = 50,
        min_abs_improvement: float = 0.03,
        metric_name: str | None = None,
        maximize: bool = False,
        smooth_window: int = 10,
        warmup: int = 10,
    ):
        self.rounds              = rounds
        self.min_abs_improvement = min_abs_improvement
        self.metric_name         = metric_name
        self.maximize            = maximize
        self.smooth_window       = smooth_window
        self.warmup          = warmup

        # Initialised here so a fresh callback instance per fit() call
        # gets the correct starting state regardless of maximize value.
        self._best_score        = -np.inf if maximize else np.inf
        self._rounds_since_best = 0

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False

        # Warmup guard — do not allow stopping before warmup.
        # Ensures the smoothing window is fully populated before any
        # score is eligible to become best_iteration.
        if epoch < self.warmup:
            return False

        eval_set = list(evals_log.keys())[-1]
        metrics  = evals_log[eval_set]
        key      = self.metric_name if self.metric_name in metrics else list(metrics.keys())[-1]
        history  = metrics[key]

        # ── Smooth over last `smooth_window` rounds before comparing ─────────
        # A single-round dip would otherwise reset patience unnecessarily.
        # The rolling average tracks the trend instead.
        window  = min(self.smooth_window, len(history))
        current = sum(history[-window:]) / window
        # ─────────────────────────────────────────────────────────────────────

        if np.isinf(self._best_score):
            threshold = 0.0  # first update after warmup always registers
        else:
            threshold = self.min_abs_improvement

        is_better = (
            current > self._best_score + threshold
            if self.maximize
            else current < self._best_score - threshold
        )

        if is_better:
            self._best_score        = current
            self._rounds_since_best = 0
        else:
            self._rounds_since_best += 1

        if self._rounds_since_best >= self.rounds:
            print(
                f"    ⏹ EarlyStoppingCallback triggered at epoch {epoch} "
                f"(no improvement > {self.min_abs_improvement} in {self.rounds} rounds, "
                f"best={self._best_score:.4f})"
            )
            return True

        return False

class XGBWrapper(BaseEstimator):
    """
    Single-target XGBoost wrapper.
    
    Since each target is trained in its own pipeline, all multi-target
    orchestration (model dicts, per-target NaN loops, averaged scoring)
    is removed. The wrapper handles one target — either regression or
    classification — with early stopping, optional log transform / scaling,
    and threshold support.

    Parameters
    ----------
    target : str
        Name of the target column.
    is_clf : bool
        True for classification, False for regression.
    model : XGBClassifier | XGBRegressor
        Unfitted model. Cloned on each fit() call.
    scale_reg_targets : bool
        Apply StandardScaler to regression targets.
    log_reg_targets : bool
        Apply signed-log transform to regression targets.
    lr_config : dict
        Maps learning_rate → {"n_estimators": int, "early_stopping_rounds": int}.
    """

    # Minimum rounds before early stopping is allowed to fire.
    # Prevents an early spike from winning before smooth_window is populated.
    WARMUP = 10
    
    _MIN_ABS_IMPROVEMENT_CLF = 0.02  # AUCPR units
    _MIN_ABS_IMPROVEMENT_REG = 0.02  # RMSE units — review per target scale
    
    _SMOOTH_WINDOW = 10

    def __init__(
        self,
        target: str,
        is_clf: bool,
        model=None,
        scale_reg_targets: bool = False,
        log_reg_targets: bool = False,
        lr_config: dict | None = None,
    ):
        self.target            = target
        self.is_clf            = is_clf
        self.model             = model
        self.scale_reg_targets = scale_reg_targets
        self.log_reg_targets   = log_reg_targets
        self.lr_config         = lr_config

        if model is None:
            raise ValueError("model must be provided")
        if lr_config is None:
            raise ValueError("lr_config must be provided")

        # Learned attributes
        self.model_   = None    # fitted model
        self.scaler_  = None    # fitted StandardScaler (regression only)
        self.threshold_ = 0.5  # tunable threshold (classification only)
        self.classes_   = None  # model.classes_ (classification only)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _log(self, y):
        """Signed log1p transform."""
        return np.sign(y) * np.log1p(np.abs(y))

    def _inv_log(self, y):
        """Inverse signed log1p."""
        return np.sign(y) * np.expm1(np.abs(y))

    def _resolve_lr_config(self):
        lr  = self.model_.get_params().get("learning_rate")
        cfg = self.lr_config.get(lr)
        if cfg is None:
            raise KeyError(
                f"learning_rate={lr} not in lr_config. "
                f"Valid: {sorted(self.lr_config.keys())}"
            )
        return cfg["n_estimators"], cfg["early_stopping_rounds"]

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #
    def fit(self, X, y, **fit_params):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        valid = ~X.isna().all(axis=1)
        X = X[valid].reset_index(drop=True)
        y = y[valid].reset_index(drop=True)

        X_val = fit_params.pop("eval_set_X", None)
        y_val = fit_params.pop("eval_set_y", None)
        if X_val is not None:
            X_val = X_val.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True) if y_val is not None else None
            valid_val = ~X_val.isna().all(axis=1)
            X_val = X_val[valid_val].reset_index(drop=True)
            if y_val is not None:
                y_val = y_val[valid_val].reset_index(drop=True)

        y_train = y[self.target] if isinstance(y, pd.DataFrame) else y
        y_val_t = (
            y_val[self.target] if (y_val is not None and isinstance(y_val, pd.DataFrame))
            else y_val
        )

        self.model_ = clone(self.model)
        n_estimators, early_stopping_rounds = self._resolve_lr_config()
        
        # Dont use XGBoosts default early stopping, custom callback handles it
        self.early_stopping_rounds_ = early_stopping_rounds
        self.model_.set_params(n_estimators=n_estimators, early_stopping_rounds=None)

        # If max_depth is 1, colsample_bynode should not be < 1
        depth = self.model_.get_params().get("max_depth", 1)
        if depth == 1:
            self.model_.set_params(colsample_bynode=1.0)

        # ---- Prepare targets before fitting ----
        if self.is_clf:
            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            self.model_.set_params(scale_pos_weight=neg / pos if pos > 0 else 1.0)

            # Use AUCPR for early stopping — tracks PR-AUC directly,
            # consistent with sklearn's average_precision_score used in score().
            self.model_.set_params(eval_metric="aucpr")
            maximize = True  # maximizing AUCPR

        else:
            if self.log_reg_targets:
                y_train = self._log(y_train)
                if y_val_t is not None:
                    y_val_t = self._log(y_val_t)
            if self.scale_reg_targets:
                self.scaler_ = StandardScaler()
                y_train = self.scaler_.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                if y_val_t is not None:
                    y_val_t = self.scaler_.transform(y_val_t.values.reshape(-1, 1)).ravel()
            maximize = False  # minimising rmse

        # ---- Early stopping callback ----
        callbacks = [
            EarlyStoppingCallback(
                # Early stopping rounds from LR config
                rounds=early_stopping_rounds,
                min_abs_improvement = self._MIN_ABS_IMPROVEMENT_CLF if self.is_clf else self._MIN_ABS_IMPROVEMENT_REG,
                maximize=maximize,
                smooth_window=self._SMOOTH_WINDOW,

                # Warmup: do not allow stopping before this many rounds.
                # Prevents a half-empty smoothing window from locking in an
                # early spike as best_iteration.
                warmup=self.WARMUP,
            )
        ]

        self.model_.set_params(callbacks=callbacks)

        if X_val is not None and y_val_t is not None:
            self.model_.fit(X, y_train, eval_set=[(X_val, y_val_t)], verbose=False)
        else:
            self.model_.fit(X, y_train, verbose=False)

        # After fit — set best_iteration from eval history so predict() uses the right trees.
        # Warmup rounds (< WARMUP) are excluded: the EarlyStoppingCallback blocks
        # early stopping during warmup, but that guard doesn't apply here.  Without
        # this slice, a noise spike in the first few rounds can become best_iteration,
        # which would make both predict() and CV score() use an undertrained model.
        # AUCPR (clf) is maximised → argmax; rmse (reg) is minimised → argmin.
        if hasattr(self.model_, "evals_result_") and self.model_.evals_result_:
            eval_key   = list(self.model_.evals_result_.keys())[-1]
            metric_key = list(self.model_.evals_result_[eval_key].keys())[-1]
            values     = self.model_.evals_result_[eval_key][metric_key]
            # Skip warmup rounds — mirror the guard in EarlyStoppingCallback.
            search_start  = min(self.WARMUP, len(values) - 1)
            search_values = values[search_start:]
            offset = (int(np.argmax(search_values)) if self.is_clf
                      else int(np.argmin(search_values)))
            best_iter = search_start + offset
            self.model_.get_booster().best_iteration = best_iter

        if self.is_clf:
            self.classes_ = self.model_.classes_

        return self

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #
    def set_threshold(self, threshold: float):
        self.threshold_ = threshold

    def predict(self, X):
        X = X.reset_index(drop=True)
        valid = ~X.isna().all(axis=1)
        result = pd.Series(np.nan, index=X.index, name=self.target)

        if valid.any():
            if self.is_clf:
                if self.threshold_ != 0.5:
                    proba = self.model_.predict_proba(X[valid])[:, 1]
                    pred  = (proba >= self.threshold_).astype(int)
                else:
                    pred = self.model_.predict(X[valid])
            else:
                pred = self.model_.predict(X[valid])
                if self.scale_reg_targets and self.scaler_ is not None:
                    pred = self.scaler_.inverse_transform(
                        pred.reshape(-1, 1)
                    ).ravel()
                if self.log_reg_targets:
                    pred = self._inv_log(pred)

            result.loc[valid] = pred

        return pd.DataFrame({self.target: result})

    def predict_proba(self, X):
        """Returns DataFrame with columns {target}_0, {target}_1."""
        if not self.is_clf:
            return None

        X = X.reset_index(drop=True)
        cols   = [f"{self.target}_{c}" for c in self.classes_]
        result = pd.DataFrame(np.nan, index=X.index, columns=cols)
        valid  = ~X.isna().all(axis=1)

        if valid.any():
            proba = self.model_.predict_proba(X[valid])
            result.loc[valid, cols] = proba

        return result

    # ------------------------------------------------------------------ #
    # Score (used by RandomizedSearchCV)                                  #
    # ------------------------------------------------------------------ #
    def score(self, X, y):
        """
        Returns Average Precision (AP) for classification via sklearn's average_precision_score.
        Note: eval_metric="aucpr" (XGBoost, trapezoidal) and this score (sklearn AP,
        right-rectangle) differ slightly. Both reward ranking positives higher;
        the gap is small but means early stopping and CV scoring optimize
        _marginally different objectives.
        Returns negative RMSE for regression.
        Both are higher-is-better so sklearn maximises correctly.
        """
        y_true = y[self.target] if isinstance(y, pd.DataFrame) else y
        y_true = y_true.reset_index(drop=True)

        y_pred_df = self.predict(X)
        y_pred = y_pred_df[self.target].reset_index(drop=True)

        mask = y_true.notna() & y_pred.notna()
        if mask.sum() == 0:
            return 0.0

        y_true_v = y_true[mask]
        y_pred_v = y_pred[mask]

        if self.is_clf:
            proba_df = self.predict_proba(X)
            pos_col  = f"{self.target}_1"
            if proba_df is not None and pos_col in proba_df.columns:
                proba_df = proba_df.reset_index(drop=True)
                proba_v = proba_df.loc[mask, pos_col]
                proba_notna = proba_v.notna()
                if proba_notna.any():
                    proba_v  = proba_v[proba_notna]
                    y_true_v = y_true_v[proba_notna]
                    if len(y_true_v.unique()) < 2:
                        return 0.0
                    try:
                        raw_ap = float(average_precision_score(y_true_v, proba_v))
                        return raw_ap
                    except Exception:
                        pass
            return 0.0
        else:
            rmse = np.sqrt(mean_squared_error(y_true_v, y_pred_v))
            return -float(rmse)

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    def evaluate_predictions(
        self,
        y_pred: pd.DataFrame,
        y_true: pd.DataFrame,
        proba_df: pd.DataFrame | None = None,
        thresholds: dict | None = None,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Compute and optionally plot evaluation metrics for this target.
        Returns a single-row metrics DataFrame consistent with the old interface.
        """
        y_pred = y_pred.reset_index(drop=True)
        y_true = y_true.reset_index(drop=True)
        if proba_df is not None:
            proba_df = proba_df.reset_index(drop=True)

        target = self.target
        if target not in y_pred.columns or target not in y_true.columns:
            return pd.DataFrame()

        mask = y_true[target].notna() & y_pred[target].notna()
        if mask.sum() == 0:
            return pd.DataFrame()

        y_true_v = y_true.loc[mask, target]
        y_pred_v = y_pred.loc[mask, target]

        if not self.is_clf:
            # ── Regression ────────────────────────────────────────────────
            finite = np.isfinite(y_true_v) & np.isfinite(y_pred_v)
            y_true_v = y_true_v[finite]
            y_pred_v = y_pred_v[finite]
            if len(y_true_v) == 0:
                return pd.DataFrame()

            row = {
                "task":     "regression",
                "target":   target,
                "rmse":     np.sqrt(mean_squared_error(y_true_v, y_pred_v)),
                "r2":       r2_score(y_true_v, y_pred_v),
                "mae":      mean_absolute_error(y_true_v, y_pred_v),
                "n_samples": int(finite.sum()),
            }
            if plot:
                self._plot_regression_diagnostics(y_true_v, y_pred_v, target)
            return pd.DataFrame([row])

        else:
            # ── Classification ────────────────────────────────────────────
            th = (thresholds or {}).get(target, self.threshold_)

            proba     = None
            auroc     = np.nan
            aucpr     = np.nan
            ap        = np.nan
            p_at_10   = np.nan
            pos_label = int(self.classes_[-1]) if self.classes_ is not None else 1

            if proba_df is not None:
                pos_col = f"{target}_1"
                if pos_col in proba_df.columns:
                    proba_full  = proba_df.loc[mask, pos_col]
                    proba_clean = proba_full[proba_full.notna()]
                    y_roc       = y_true_v[proba_full.notna()]

                    if len(y_roc) >= 2 and len(np.unique(y_roc)) == 2:
                        auroc = roc_auc_score(y_roc, proba_clean)
                        precision, recall, _ = precision_recall_curve(
                            y_roc, proba_clean, pos_label=pos_label)
                        aucpr = auc(recall, precision)
                        ap = average_precision_score(y_roc, proba_clean, pos_label=pos_label)
                    # Precision@Top10
                    k       = 10
                    top_idx = np.argsort(proba_clean.values)[-k:]
                    p_at_10 = np.sum(y_roc.iloc[top_idx] == pos_label) / len(top_idx)

                    proba    = proba_clean
                    y_pred_v = (proba_full >= th).astype(int)

            # Normalized AP = (AP - prevalence) / (1 - prevalence)
            prevalence = (y_true_v == pos_label).mean()
            norm_ap    = (float(ap) - float(prevalence)) / (1.0 - float(prevalence)) if (not np.isnan(ap) and prevalence < 1.0) else np.nan
            
            accuracy      = accuracy_score(y_true_v, y_pred_v)
            recall_pos    = recall_score(y_true_v, y_pred_v, pos_label=pos_label, zero_division=0)
            precision_pos = precision_score(y_true_v, y_pred_v, pos_label=pos_label, zero_division=0)
            f1_pos        = f1_score(y_true_v, y_pred_v, pos_label=pos_label, zero_division=0)
            prec_over_base = precision_pos / prevalence if prevalence > 0 else np.nan

            row = {
                "task":                        "classification",
                "target":                      target,
                "prevalence":                  float(prevalence),
                "accuracy":                    accuracy,
                "recall_pos":                  recall_pos,
                "precision_pos":               precision_pos,
                "f1_pos":                      f1_pos,
                "auroc":                       auroc,
                "aucpr":                       aucpr,
                "ap":                          ap,
                "norm_ap":                     norm_ap,
                "precision_pos_at_top10":      p_at_10,
                "precision_pos_over_baseline": prec_over_base,
                "n_samples":                   int(mask.sum()),
            }
            if plot:
                self._plot_classification_diagnostics(
                    y_true_v, y_pred_v, target, proba=proba
                )
            return pd.DataFrame([row])

    # ------------------------------------------------------------------ #
    # Plotting (kept identical to XGBWrapper)                     #
    # ------------------------------------------------------------------ #
    def plot_learning_curves(self, figsize=(8, 4)):
        if not hasattr(self.model_, "evals_result_") or not self.model_.evals_result_:
            print(f"  (no eval history for {self.target})")
            return

        fig, ax = plt.subplots(figsize=figsize)
        for dataset, metrics in self.model_.evals_result_.items():
            for metric, values in metrics.items():
                if metric.lower() == "aucpr":
                    ax.plot(values, label=f"{dataset} {metric}")

        best        = self.model_.get_booster().best_iteration
        values_list = next(iter(next(iter(self.model_.evals_result_.values())).values()))
        last_round  = len(values_list) - 1

        ax.axvline(best,       color="red",    linestyle="--", lw=1, label=f"best={best}")
        ax.axvline(last_round, color="orange", linestyle=":",  lw=1, label=f"stopped={last_round}")

        ax.set_title(self.target)
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Metric")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.show()

    def _plot_regression_diagnostics(self, y_true, y_pred, target):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        residuals = y_true - y_pred
        if self.log_reg_targets:
            yt_log = np.sign(y_true) * np.log1p(np.abs(y_true))
            yp_log = np.sign(y_pred) * np.log1p(np.abs(y_pred))
            log_lbl = "signed log"
        else:
            yt_log, yp_log, log_lbl = y_true, y_pred, "original"
        log_res = yt_log - yp_log

        for row_data, col_data, ax, color, xlabel, ylabel, title in [
            (y_true,  y_pred,  axes[0,0], "skyblue", "True",          "Predicted",    f"{target}: Predictions (Normal)"),
            (yt_log,  yp_log,  axes[0,1], "orange",  f"True ({log_lbl})", f"Pred ({log_lbl})", f"{target}: Predictions (Log)"),
            (y_pred,  residuals, axes[1,0], "skyblue", "Predicted",    "Residual",     f"{target}: Residuals (Normal)"),
            (yp_log,  log_res,  axes[1,1], "orange",  f"Pred ({log_lbl})", "Residual", f"{target}: Residuals (Log)"),
        ]:
            axes_obj = ax
            axes_obj.scatter(row_data, col_data, alpha=0.4, color=color, s=10)
            if "Residual" in title:
                axes_obj.axhline(0, color="r", linestyle="--", lw=1)
            else:
                lo = min(row_data.min(), col_data.min())
                hi = max(row_data.max(), col_data.max())
                axes_obj.plot([lo, hi], [lo, hi], "r--", lw=1)
            axes_obj.set_xlabel(xlabel)
            axes_obj.set_ylabel(ylabel)
            axes_obj.set_title(title)
            axes_obj.grid(True, alpha=0.3)

        for data, log_data, ax, label in [
            (y_true, yt_log, axes[2,0], "Normal"),
            (y_true, yt_log, axes[2,1], "Log"),
        ]:
            d = data if label == "Normal" else log_data
            p = y_pred if label == "Normal" else yp_log
            sns.kdeplot(d, fill=True, color="skyblue", alpha=0.5, bw_adjust=0.2,
                        cut=0, label="True", ax=ax)
            sns.kdeplot(p, fill=True, color="orange",  alpha=0.5, bw_adjust=0.2,
                        cut=0, label="Predicted", ax=ax)
            ax.set_title(f"{target}: Distribution ({label})")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Regression Diagnostics: {target}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def _plot_classification_diagnostics(self, y_true, y_pred, target, proba=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
        axes[0].set_title(f"{target}: Confusion Matrix")

        unique = np.unique(np.concatenate([y_true, y_pred]))
        x, w = np.arange(len(unique)), 0.35
        axes[1].bar(x - w/2, [np.sum(y_true == c) for c in unique], w,
                    label="True", color="skyblue", alpha=0.7)
        axes[1].bar(x + w/2, [np.sum(y_pred == c) for c in unique], w,
                    label="Predicted", color="orange", alpha=0.7)
        axes[1].set_xticks(x); axes[1].set_xticklabels(unique)
        axes[1].set_title(f"{target}: Class Distribution")
        axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

        if proba is not None:
            fpr, tpr, _ = roc_curve(y_true, proba)
            axes[2].plot(fpr, tpr, "b-", lw=2, label=f"AUC={auc(fpr,tpr):.3f}")
            axes[2].plot([0,1], [0,1], "r--", lw=1)
            axes[2].set_title(f"{target}: ROC Curve"); axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, "No probabilities", ha="center", va="center",
                         transform=axes[2].transAxes)
            axes[2].set_title(f"{target}: ROC (unavailable)")

        plt.suptitle(f"Classification Diagnostics: {target}", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()

# ============================================
# SPLITTER
# ============================================
class MultiTargetTimeSeriesSplitter:
    """
    Moving-window time series splitter yielding (train, val, test) or (train, val) index
    arrays per fold.

    Window layout within each fold
    --------------------------------
    |<--- train --->|<-- val -->|<-- gap -->|<-- test -->|
                                ^^^^^^^^^^^
                         gap_years of dead-zone rows
                         are excluded from val (shrinks
                         val from the right) so that no
                         val observation falls within
                         gap_years of the test period.

    Data-efficiency design
    ----------------------
    Folds advance by `step` rows (default = test_size).  Using step = test_size
    guarantees non-overlapping test sets (required for independent evaluation)
    while maximising overlap of the training windows across folds — the most
    data-efficient valid configuration.

    Window size is derived so the last fold ends exactly at the last row:

        window_size = total_len - (n_splits - 1) * step

    which is always >= actual_window_size implied by the proportions, so the
    proportions are applied *within* each window.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    date_column : str
        Name of the datetime column in X.
    gap_years : int
        Minimum number of years between the last val observation and the first
        test observation.  Rows in the gap are simply dropped from val (val
        shrinks from the right); train is never touched.
    train_prop : float
        Share of the window assigned to training.
    val_prop : float
        Share of the window assigned to validation.
    test_prop : float
        Share of the window assigned to test (0 → yield only train/val).
    step : int | None
        How many rows to advance between consecutive folds.  Defaults to
        test_size (non-overlapping tests, maximum train reuse).
    """

    def __init__(
        self,
        n_splits: int = 2,
        date_column: str = None,
        gap_years: int = 0,
        train_prop: float = 0.5,
        val_prop: float = 0.25,
        test_prop: float = 0.25,
        step: int | None = None,
    ):
        if not (0 < train_prop < 1 and 0 < val_prop < 1 and 0 <= test_prop < 1):
            raise ValueError(
                "train_prop and val_prop must be in (0, 1); test_prop in [0, 1)."
            )
        total = train_prop + val_prop + test_prop
        if not abs(total - 1.0) < 1e-9:
            raise ValueError(f"Proportions must sum to 1.0, got {total:.6f}.")
        if test_prop == 0 and gap_years > 0:
            raise ValueError("gap_years requires test_prop > 0.")

        self.n_splits = n_splits
        self.date_column = date_column
        self.gap_years = gap_years
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop
        self._step_override = step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_window_params(self, total_len: int) -> tuple[int, int, int, int, int]:
        """
        Return (window_size, train_size, val_size, test_size, step).

        Strategy
        --------
        step and window_size are coupled:
            step        = test_size = floor(window_size * test_prop)
            window_size = total_len - (n_splits - 1) * step

        Substituting and solving analytically:
            step = floor(total_len * test_prop / (1 + (n_splits - 1) * test_prop))

        This guarantees step == test_size so test windows never overlap.
        """
        if self.n_splits == 1:
            window_size = total_len
            step = max(1, int(window_size * self.test_prop)) if self.test_prop > 0 else total_len
        else:
            if self._step_override is not None:
                step = self._step_override
            else:
                if self.test_prop > 0:
                    # Closed-form solution: step == test_size by construction.
                    step = max(1, int(
                        (total_len * self.test_prop) /
                        (1 + (self.n_splits - 1) * self.test_prop)
                    ))
                else:
                    # No test set: advance by val_size instead.
                    step = max(1, int(
                        (total_len * self.val_prop) /
                        (1 + (self.n_splits - 1) * self.val_prop)
                    ))

            # Back-compute window so the last fold ends exactly at total_len.
            window_size = total_len - (self.n_splits - 1) * step

        if window_size <= 0:
            raise ValueError(
                f"Computed window_size={window_size} <= 0. "
                f"Reduce n_splits or provide more data."
            )

        train_size = max(1, int(window_size * self.train_prop))
        val_size   = max(1, int(window_size * self.val_prop))
        test_size  = max(1, int(window_size * self.test_prop)) if self.test_prop > 0 else 0

        # Use all window rows (avoids silent remainder waste).
        # Assign leftover to train (largest, least sensitive to small changes).
        used = train_size + val_size + test_size
        leftover = window_size - used
        train_size += leftover

        if self.n_splits == 1:
            step = test_size if test_size > 0 else val_size

        return window_size, train_size, val_size, test_size, step

    def _apply_gap(
        self,
        val_start: int,
        val_end: int,
        test_start: int,
        dates: pd.Series,
    ) -> int:
        """
        Shrink val_end so no val date is within gap_years of the first test date.
        Train is never modified.  Returns adjusted val_end.
        """
        if self.gap_years == 0 or test_start >= len(dates):
            return val_end

        first_test_date = dates.iloc[test_start]
        cutoff = first_test_date - pd.DateOffset(years=self.gap_years)
        # Last row index (in the valid-only frame) whose date is < cutoff.
        new_val_end = int(dates.searchsorted(cutoff, side="right"))
        # Never push val_end below val_start (would create empty val).
        return max(val_start, min(val_end, new_val_end))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self, X: pd.DataFrame, y=None, groups=None
    ) -> Iterator[tuple]:
        """Yield index arrays for each fold."""
        if self.date_column is None:
            raise ValueError("date_column must be specified.")

        dates = pd.to_datetime(X[self.date_column])
        valid_mask = dates.notna()
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid (non-NaT) dates found in the data.")

        dates_valid = dates.iloc[valid_indices].reset_index(drop=True)
        total_len = len(valid_indices)

        window_size, train_size, val_size, test_size, step = self._compute_window_params(
            total_len
        )

        actual_folds = 0
        for fold_idx in range(self.n_splits):
            window_start = fold_idx * step
            window_end   = window_start + window_size

            if window_end > total_len:
                import warnings
                warnings.warn(
                    f"Fold {fold_idx} would exceed data length ({window_end} > {total_len}). "
                    f"Stopping at {actual_folds} folds.",
                    UserWarning,
                    stacklevel=2,
                )
                break

            train_start = window_start
            train_end   = train_start + train_size
            val_start   = train_end
            val_end     = val_start + val_size
            test_start  = val_end
            test_end    = test_start + test_size

            # Gap: only shrinks val from the right — train is never touched.
            if self.test_prop > 0:
                val_end = self._apply_gap(val_start, val_end, test_start, dates_valid)

            train_idx = valid_indices[train_start:train_end]
            val_idx   = valid_indices[val_start:val_end]

            if len(train_idx) == 0:
                raise RuntimeError(f"Fold {fold_idx}: empty train set — reduce n_splits.")
            if len(val_idx) == 0:
                raise RuntimeError(
                    f"Fold {fold_idx}: empty val set after gap removal — "
                    "reduce gap_years or increase val_prop."
                )

            if self.test_prop > 0:
                test_idx = valid_indices[test_start:test_end]
                if len(test_idx) == 0:
                    raise RuntimeError(f"Fold {fold_idx}: empty test set — reduce n_splits.")
                yield train_idx, val_idx, test_idx
            else:
                yield train_idx, val_idx

            actual_folds += 1

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a human-readable DataFrame showing date ranges and sizes per fold.
        Useful for sanity-checking before running a full CV loop.
        """
        if self.date_column is None:
            raise ValueError("date_column must be specified.")

        dates = pd.to_datetime(X[self.date_column])
        rows = []

        for fold_idx, splits in enumerate(self.split(X)):
            if self.test_prop > 0:
                train_idx, val_idx, test_idx = splits
            else:
                train_idx, val_idx = splits
                test_idx = np.array([], dtype=int)

            def _range(idx):
                if len(idx) == 0:
                    return "—", "—"
                return str(dates.iloc[idx[0]].date()), str(dates.iloc[idx[-1]].date())

            tr_s, tr_e = _range(train_idx)
            v_s,  v_e  = _range(val_idx)
            te_s, te_e = _range(test_idx)

            rows.append({
                "fold":        fold_idx,
                "train_start": tr_s, "train_end": tr_e, "train_n": len(train_idx),
                "val_start":   v_s,  "val_end":   v_e,  "val_n":   len(val_idx),
                "test_start":  te_s, "test_end":  te_e, "test_n":  len(test_idx),
            })

        return pd.DataFrame(rows)

# ============================================
# EARLY STOPPING PIPELINE (allows for transforming val set to tune early stopping)
# ============================================

class EarlyStoppingPipeline(Pipeline):
    """Pipeline that transforms validation data using fitted steps."""

    def fit(self, X, y=None, **fit_params):
        # Extract validation sets if provided
        eval_set_X = fit_params.pop('eval_set_X', None)
        eval_set_y = fit_params.pop('eval_set_y', None)

        # Fit all steps except the final estimator
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                Xt = step.fit_transform(Xt, y)
            elif hasattr(step, 'fit'):
                step.fit(Xt, y)
                Xt = step.transform(Xt) if hasattr(step, 'transform') else Xt
            else:
                Xt = step.transform(Xt) if hasattr(step, 'transform') else Xt

        # If validation data is provided, transform it through the same steps
        if eval_set_X is not None:
            X_val_transformed = eval_set_X
            for name, step in self.steps[:-1]:
                if hasattr(step, 'transform'):
                    X_val_transformed = step.transform(X_val_transformed)
            fit_params['eval_set_X'] = X_val_transformed
            fit_params['eval_set_y'] = eval_set_y  # y_val is not transformed (targets unchanged)

        # Fit the final estimator with (possibly) transformed validation data
        self.steps[-1][1].fit(Xt, y, **fit_params)
        return self

# ============================================
# Tunable Pipeline Factory
# ============================================

class TunablePipelineFactory:
    """
    Pipeline order
    --------------
    1. get_parsing_pipeline()    – run on the FULL dataset before splitting.
    2. get_binarizer_pipeline()  – run per fold, fit on train only (no leakage).
    3. get_tunable_pipeline()    – model pipeline, used inside CV.
    """

    def __init__(
        self,
        # Core configuration
        cat_columns,
        cat_list_columns,
        count_list_columns,
        count_lol_columns,
        columns_to_drop_2,

        # Optional configurations
        remove_cat=None,
        remove_string=None,
        remove_cat_with_string=None,

        # Objectives
        reg_objective="reg:squarederror",
        clf_objective="binary:logistic",

        # Options
        scale_reg_targets=False,
        log_reg_targets=False,

        # Paths and files
        tby_file_path="data/macroeconomy/10_year_bond_yield_clean.csv",
        tby_columns_to_merge=None,
        embedding_paths=None,
        embedding_model=None,
        scoring_paths=None,
        scoring_model=None,
        memory=None,
        cache_location="data/pipeline/cache",

        # Preprocessed data required for numeric features
        preprocessed_data=None,
    ):
        if preprocessed_data is None:
            raise ValueError("preprocessed_data is required to compute numeric features")
        self.preprocessed_data = preprocessed_data

        self.cat_columns        = cat_columns
        self.cat_list_columns   = cat_list_columns
        self.count_list_columns = count_list_columns
        self.count_lol_columns  = count_lol_columns
        self.columns_to_drop_2  = columns_to_drop_2

        self.remove_cat            = remove_cat
        self.remove_string         = remove_string
        self.remove_cat_with_string = remove_cat_with_string

        self.reg_objective = reg_objective
        self.clf_objective = clf_objective

        self.scale_reg_targets = scale_reg_targets
        self.log_reg_targets   = log_reg_targets

        self.tby_file_path       = tby_file_path
        self.tby_columns_to_merge = tby_columns_to_merge or {
            "US 10 Year": "ten_year_tby_us",
            "EU 10 Year": "ten_year_tby_eu",
        }
        self.embedding_paths  = embedding_paths
        self. embedding_model = embedding_model
        self.scoring_paths    = scoring_paths
        self.scoring_model    = scoring_model
        self.embedding_pca_dim = 100

        if memory is None:
            self.memory = Memory(location=cache_location, verbose=0)
        else:
            self.memory = memory

    # --------------------------------------------------
    # 1. Pre-split parsing pipeline (run on full dataset)
    # --------------------------------------------------
    def get_parsing_pipeline(
        self,
        fx_cache_file,
        buyin_round,
        window_years,
        download_date,
        exit_round_types,
    ):
        """
        Stateless preprocessing pipeline — safe to run on the full dataset
        before the train/test split.

        Expands events, attaches regression targets, and censors rows whose
        observation window extends beyond download_date. Binary quantile flags
        are NOT created here; use get_binarizer_pipeline() after splitting.
        """
        return Pipeline([
            ("feature_extractor", FeatureExtractor()),
            ("buyin_round_extractor", BuyInRoundExtractor(
                fx_cache_file=fx_cache_file,
                buyin_round_number=buyin_round,
            )),
            ("target_extractor", TargetExtractor(
                fx_cache_file=fx_cache_file,
                window_years=window_years,
                download_date=download_date,
                exit_round_types=exit_round_types,
            )),
        ], memory=self.memory, verbose=False)

    # --------------------------------------------------
    # 2. Binarizer pipeline (used inside CV but before tunable pipeline, which only takes features)
    # --------------------------------------------------
    def get_binarizer_pipeline(self, binary_targets=None, quantile_threshold=0.75):
        return Pipeline([
            ("quantile_binarizer", QuantileTargetBinarizer(
                binary_targets=binary_targets,
                quantile_threshold=quantile_threshold,
            )),
        ], verbose=False)  # No memory — must refit per fold

    # --------------------------------------------------
    # 3. Tunable model pipeline (used inside CV)
    # --------------------------------------------------
    def get_tunable_pipeline(
        self,
        target: str,
        is_clf: bool,
        lr_config: dict,
        encode_cat: bool = True,
        random_state: int = 42,
    ):
        xgb_reg = XGBRegressor(
            objective=self.reg_objective,
            random_state=random_state,
            verbosity=0,
            n_jobs=1,
        )
        xgb_clf = XGBClassifier(
            objective=self.clf_objective,
            random_state=random_state,
            verbosity=0,
            n_jobs=1,
        )

        steps = [
            ("target_column_dropper", TargetColumnDropper()),
            ("embedding_score_merger", EmbeddingScoreMerger(
                input_type="score",
                embedding_paths=self.embedding_paths,
                embedding_model=self.embedding_model,
                embedding_pca_dim=self.embedding_pca_dim,
                scoring_paths=self.scoring_paths,
                scoring_model=self.scoring_model,
            )),
            ("ten_year_tby_merger", TenYearTBYMerger(
                file_path=self.tby_file_path,
                columns_to_merge=self.tby_columns_to_merge,
                merge=True,
            )),
            ("column_dropper_2", ColumnDropper(cols_to_drop=self.columns_to_drop_2)),
        ]

        if encode_cat:
            all_cat_cols = (
                self.cat_columns
                + self.cat_list_columns
                + self.count_list_columns
                + self.count_lol_columns
            )
            encoder = ColumnTransformer(
                transformers=[(
                    "cat_encoder",
                    CatEncoder(
                        cat_columns=self.cat_columns,
                        cat_list_columns=self.cat_list_columns,
                        count_list_columns=self.count_list_columns,
                        count_lol_columns=self.count_lol_columns,
                        remove_cat=self.remove_cat,
                        remove_string=self.remove_string,
                        remove_cat_with_string=self.remove_cat_with_string,
                        min_frequency=10,
                    ),
                    all_cat_cols,
                )],
                remainder="passthrough",
            )
            encoder.set_output(transform="pandas")
            steps.append(("encoder", encoder))

        steps.append(("xgb_wrapper", XGBWrapper(
            target=target,
            is_clf=is_clf,
            model=xgb_clf if is_clf else xgb_reg,
            scale_reg_targets=self.scale_reg_targets,
            log_reg_targets=self.log_reg_targets,
            lr_config=lr_config,
        )))

        return EarlyStoppingPipeline(steps, verbose=False)