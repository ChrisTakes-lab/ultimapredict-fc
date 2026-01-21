import pandas as pd
import requests
import os
import glob
import streamlit as st

# ───────────────────────────────────────────────
# Try to load secrets – fallback to empty strings if not found
FOOTBALL_DATA_KEY = st.secrets.get("FOOTBALL_DATA_KEY", "")
API_FOOTBALL_KEY  = st.secrets.get("API_FOOTBALL_KEY",  "")
APIFOOTBALL_KEY   = st.secrets.get("APIFOOTBALL_KEY",   "")
ODDS_API_KEY      = st.secrets.get("ODDS_API_KEY",      "")
TELEGRAM_TOKEN    = st.secrets.get("TELEGRAM_TOKEN",    "")
TELEGRAM_CHAT_ID  = st.secrets.get("TELEGRAM_CHAT_ID",  "")

# ───────────────────────────────────────────────
def load_local_csvs():
    """
    Look for any .csv files the user manually placed in a 'historical_data/' folder
    """
    folder = "historical_data"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return pd.DataFrame()

    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        st.info("No CSV files found in historical_data/ folder.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            st.warning(f"Could not read {os.path.basename(f)} → {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    st.success(f"Loaded {len(combined)} rows from {len(dfs)} local CSV file(s)")
    return combined


def download_csv_fallback():
    """
    Download E0.csv files from football-data.co.uk – with heavy safety
    """
    base_url = "https://www.football-data.co.uk/mmz4281/"
    seasons = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2014, 2027)]

    dfs = []
    for season in seasons:
        url = f"{base_url}{season}/E0.csv"
        try:
            df = pd.read_csv(url, on_bad_lines='skip', low_memory=False)
            if df.empty:
                continue
            if 'Date' not in df.columns:
                possible = [c for c in df.columns if 'date' in c.lower()]
                if possible:
                    df = df.rename(columns={possible[0]: 'Date'})
            dfs.append(df)
            st.write(f"✓ Downloaded {len(df)} matches – {season}")
        except Exception as e:
            st.write(f"✗ Failed {season}: {str(e)[:80]}...")

    if not dfs:
        st.error("Could not download any historical data from football-data.co.uk")
        return pd.DataFrame()

    full = pd.concat(dfs, ignore_index=True)
    return full


def fetch_historical():
    """
    1. Try local user-provided CSVs first
    2. Then try downloading fresh data
    3. Return minimal dummy if everything fails
    """
    # Priority 1: user's manual CSVs
    df_local = load_local_csvs()
    if not df_local.empty:
        return df_local

    # Priority 2: download
    df_web = download_csv_fallback()
    if not df_web.empty:
        return df_web

    # Last resort: tiny dummy so the app doesn't crash
    st.warning("Using minimal dummy data – predictions will be very limited")
    dummy = pd.DataFrame({
        'Date':      ['2025-08-10', '2025-08-16', '2025-08-23'],
        'HomeTeam':  ['Arsenal',    'Man City',   'Liverpool'],
        'AwayTeam':  ['Wolves',     'Chelsea',    'Brighton'],
        'FTHG':      [2,            3,            1],
        'FTAG':      [1,            1,            1],
        'FTR':       ['H',          'D',          'A']
    })
    dummy['Date'] = pd.to_datetime(dummy['Date'])
    return dummy


def process_data(df):
    """
    Safe processing – never assume columns exist
    """
    if df is None or df.empty:
        st.warning("No data available → empty DataFrame returned")
        return pd.DataFrame(columns=['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'])

    # Show what we actually received
    st.write("Columns found in data:", list(df.columns))

    # Find date column (case insensitive)
    date_candidates = [c for c in df.columns if 'date' in c.lower()]
    date_col = date_candidates[0] if date_candidates else None

    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            df = df.sort_values(date_col)
        except Exception as e:
            st.warning(f"Could not parse dates from '{date_col}': {e}")
    else:
        st.warning("No date column detected → skipping date sorting")

    # Standardize column names we care about
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'home' in cl and 'team' in cl:   rename_map[c] = 'HomeTeam'
        if 'away' in cl and 'team' in cl:   rename_map[c] = 'AwayTeam'
        if 'fthg' in cl or 'home goals' in cl: rename_map[c] = 'FTHG'
        if 'ftag' in cl or 'away goals' in cl: rename_map[c] = 'FTAG'
        if 'ftr'  in cl or 'result' in cl:     rename_map[c] = 'FTR'

    df = df.rename(columns=rename_map)

    # Keep only columns we actually use downstream
    keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    existing = [c for c in keep if c in df.columns]
    df = df[existing]

    st.write(f"Processed data shape: {df.shape}")
    return df


# Keep your other functions (get_odds, fetch_fixtures, etc.) unchanged
# ───────────────────────────────────────────────
