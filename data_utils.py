import pandas as pd
import requests
import os
import glob
import streamlit as st

# Load secrets safely
FOOTBALL_DATA_KEY = st.secrets.get("FOOTBALL_DATA_KEY", "")
API_FOOTBALL_KEY  = st.secrets.get("API_FOOTBALL_KEY",  "")
APIFOOTBALL_KEY   = st.secrets.get("APIFOOTBALL_KEY",   "")
ODDS_API_KEY      = st.secrets.get("ODDS_API_KEY",      "")
TELEGRAM_TOKEN    = st.secrets.get("TELEGRAM_TOKEN",    "")
TELEGRAM_CHAT_ID  = st.secrets.get("TELEGRAM_CHAT_ID",  "")

def load_local_csvs():
    folder = "historical_data"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return pd.DataFrame()

    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        st.info("No CSV files in historical_data/ – using web fallback.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to read {os.path.basename(f)}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    st.success(f"Loaded {len(combined)} rows from local CSVs")
    return combined


def download_csv_fallback():
    base_url = "https://www.football-data.co.uk/mmz4281/"
    seasons = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2014, 2027)]
    dfs = []

    for season in seasons:
        url = f"{base_url}{season}/E0.csv"
        try:
            df = pd.read_csv(url, on_bad_lines='skip', low_memory=False)
            if df.empty or 'Date' not in df.columns:
                continue
            dfs.append(df)
            st.write(f"Loaded {len(df)} matches from {season}")
        except Exception as e:
            st.write(f"Failed {season}: {str(e)[:60]}...")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def fetch_historical():
    local_df = load_local_csvs()
    if not local_df.empty:
        return local_df

    web_df = download_csv_fallback()
    if not web_df.empty:
        return web_df

    st.warning("No data sources worked – using tiny dummy")
    dummy = pd.DataFrame({
        'Date': pd.to_datetime(['2025-08-10', '2025-08-16']),
        'HomeTeam': ['Arsenal', 'Man City'],
        'AwayTeam': ['Wolves', 'Chelsea'],
        'FTHG': [2, 3],
        'FTAG': [1, 1],
        'FTR': ['H', 'D']
    })
    return dummy


def process_data(df):
    if df.empty:
        return df

    st.write("Raw columns:", list(df.columns))

    # Find date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.sort_values(date_col)

    # Rename standard columns
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if 'home' in cl and 'team' in cl: rename[c] = 'HomeTeam'
        if 'away' in cl and 'team' in cl: rename[c] = 'AwayTeam'
        if 'fthg' in cl: rename[c] = 'FTHG'
        if 'ftag' in cl: rename[c] = 'FTAG'
        if 'ftr' in cl: rename[c] = 'FTR'

    df = df.rename(columns=rename)

    keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    existing = [c for c in keep if c in df.columns]
    return df[existing]


def fetch_fixtures(league_id=39, season=2025, upcoming=True):
    # Placeholder – implement your API logic here later
    # For now return empty to avoid further errors
    st.warning("fetch_fixtures not fully implemented yet – returning empty")
    return pd.DataFrame()


def get_real_time_stats(team_name, league_id=39, season=2025):
    # Placeholder
    st.warning("get_real_time_stats placeholder")
    return {'gf': 0, 'ga': 0, 'form': 'N/A'}
