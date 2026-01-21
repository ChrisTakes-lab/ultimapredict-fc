import pandas as pd
import requests
import os
import streamlit as st  # For secrets

# Load keys from Streamlit secrets (add in Step 7)
FOOTBALL_DATA_KEY = st.secrets.get("FOOTBALL_DATA_KEY", "fallback")
API_FOOTBALL_KEY = st.secrets.get("API_FOOTBALL_KEY", "fallback")
APIFOOTBALL_KEY = st.secrets.get("APIFOOTBALL_KEY", "fallback")
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "fallback")
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "fallback")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "fallback")

def download_csv_fallback():
    os.makedirs('data', exist_ok=True)
    years = range(2015, 2026)
    dataframes = []
    for y in years:
        url = f"https://www.football-data.co.uk/mmz4281/{y % 100:02d}{(y + 1) % 100:02d}/E0.csv"
        try:
            df = pd.read_csv(url, on_bad_lines='skip')
            dataframes.append(df)
        except:
            pass
    full_df = pd.concat(dataframes, ignore_index=True)
    full_df.to_csv('data/historical.csv', index=False)
    return full_df

def fetch_fixtures(league_id=39, season=2025, upcoming=True):  # api-football.com (PL=39)
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    status = "NS,TBD" if upcoming else "FT"
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}&status={status}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        fixtures = response.json()['response']
        return pd.DataFrame([{
            'fixture_id': f['fixture']['id'],
            'Date': f['fixture']['date'],
            'HomeTeam': f['teams']['home']['name'],
            'AwayTeam': f['teams']['away']['name'],
            'FTR': f['goals']['home'] > f['goals']['away'] and 'H' or f['goals']['home'] < f['goals']['away'] and 'A' or 'D' if not upcoming else None
        } for f in fixtures])
    # Fallback to football-data.org
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"http://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED" if upcoming else "..."
    # ... similar parsing
    # Extra fallback: apifootball.com
    url = f"https://apiv3.apifootball.com/?action=get_events&from={season}-01-01&to={season+1}-12-31&league_id=152&APIkey={APIFOOTBALL_KEY}"
    # Parse JSON to DF
    return download_csv_fallback()  # Ultimate fallback

def fetch_historical(league_id=39, season=2025):
    # Similar to fixtures but status=FT, multi-season aggregate
    df = pd.DataFrame()
    for s in range(2015, season+1):
        temp = fetch_fixtures(league_id, s, upcoming=False)
        df = pd.concat([df, temp])
    # Augment with StatsBomb open data (free GitHub)
    sb_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/37/{season}.json"  # PL code 37
    sb_resp = requests.get(sb_url.replace('{season}', str(season - 2000)))  # Adjust
    if sb_resp.status_code == 200:
        # Parse xG, events to features
        pass  # Add columns like 'xG_home'
    return df

def get_real_time_stats(team_name, league_id=39, season=2025):
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    team_url = f"https://v3.football.api-sports.io/teams?search={team_name}&league={league_id}&season={season}"
    team_id = requests.get(team_url, headers=headers).json()['response'][0]['team']['id']
    stats_url = f"https://v3.football.api-sports.io/teams/statistics?team={team_id}&league={league_id}&season={season}"
    resp = requests.get(stats_url, headers=headers)
    if resp.status_code == 200:
        stats = resp.json()['response']
        return {'gf': stats['goals']['for']['total']['total'], 'ga': stats['goals']['against']['total']['total'], 'form': stats['form']}
    # Fallback FBref scrape
    from bs4 import BeautifulSoup
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    # Parse table for team stats
    return {'gf': 40, 'ga': 20, 'form': 'WWDLW'}  # Demo

def get_odds(home, away, markets='h2h,totals'):
    # The Odds API with your key
    sport = 'soccer_epl'  # Expand to others
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={ODDS_API_KEY}&regions=us,eu,uk,au&markets={markets}&oddsFormat=decimal&bookmakers=draftkings,fanduel,betmgm,bet365,pinnacle"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        match_odds = next((event for event in data if (event['home_team'] in home or home in event['home_team']) and (event['away_team'] in away or away in event['away_team'])), None)
        if match_odds:
            averages = {'home_win': 0, 'draw': 0, 'away_win': 0, 'over_2_5': 0, 'bookmakers_count': 0}
            for bm in match_odds['bookmakers']:
                averages['bookmakers_count'] += 1
                for m in bm['markets']:
                    if m['key'] == 'h2h':
                        for o in m['outcomes']:
                            if o['name'] == home: averages['home_win'] += o['price']
                            elif o['name'] == 'Draw': averages['draw'] += o['price']
                            elif o['name'] == away: averages['away_win'] += o['price']
                    if m['key'] == 'totals' and any(o['point'] == 2.5 for o in m['outcomes']):
                        for o in m['outcomes']:
                            if o['name'] == 'Over' and o['point'] == 2.5: averages['over_2_5'] += o['price']
            for k in ['home_win', 'draw', 'away_win', 'over_2_5']:
                if averages['bookmakers_count'] > 0:
                    averages[k] /= averages['bookmakers_count']
            return averages
    # Fallback scrape (oddschecker or similar)
    from bs4 import BeautifulSoup
    scrape_url = f"https://www.oddschecker.com/football/english/premier-league/{home.lower().replace(' ', '-')}-v-{away.lower().replace(' ', '-')}/winner"
    resp = requests.get(scrape_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Extract odds from DOM (simplified; use class selectors)
    return {'home_win': 5.5, 'draw': 4.0, 'away_win': 1.6, 'over_2_5': 1.8}  # Example decimals

def get_external_prediction(fixture_id):
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    url = f"https://v3.football.api-sports.io/predictions?fixture={fixture_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        pred = resp.json()['response'][0]['predictions']
        return {
            'win_prob': {'Home Win': float(pred['percent']['home'].strip('%')), 'Draw': float(pred['percent']['draw'].strip('%')), 'Away Win': float(pred['percent']['away'].strip('%'))},
            'over_25': float(pred['under_over']['+2.5'].strip('%')) if '+2.5' in pred['under_over'] else 50
        }
    return {'win_prob': {'Home Win': 50, 'Draw': 0, 'Away Win': 50}, 'over_25': 50}  # Fallback

def process_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    # Feature eng: Elo, rolling form, xG from StatsBomb
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    df['home_elo'] = 1500  # Init, update with logic
    # ... add more
    return df
