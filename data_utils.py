# data_utils.py
# The foundation: data fetching, odds, real-time stats, examples — no ML/model awareness

import pandas as pd
import numpy as np
# Add your real imports: requests, json, datetime, etc. for live data/odds
# e.g. import requests, from datetime import datetime, timedelta

def get_examples():
    """Return sample input data for demo/prediction preview"""
    # Your real logic — placeholder example
    data = {
        'match_id': [1, 2, 3],
        'team_home': ['Man Utd', 'Real Madrid', 'Bayern'],
        'team_away': ['Liverpool', 'Barcelona', 'Dortmund'],
        'odds_home': [2.1, 1.8, 2.5],
        # ... more features
    }
    return pd.DataFrame(data)

def get_odds(match_id=None, league=None):
    """Fetch current odds — API, scrape, or static"""
    # Your real odds source (e.g. API call to Pinnacle/Bet365 wrapper, or cached CSV)
    # Placeholder
    return {
        'home': 2.10,
        'draw': 3.40,
        'away': 3.20,
        'over_2.5': 1.95
    }

def get_real_time_stats(team_home, team_away, current_time=None):
    """Live/in-game stats: possession, shots, xG, etc."""
    # Your real-time source (e.g. websocket, API poll, or simulated)
    # Placeholder
    return {
        'possession_home': 58,
        'shots_home': 7,
        'xg_home': 1.2,
        # ...
    }

# Add any other pure data functions: load_historical, preprocess_features, etc.
# Keep this file clean — NO imports from model/
