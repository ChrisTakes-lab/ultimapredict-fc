import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from telegram import Bot

# ───────────────────────────────────────────────
# Debug import block – remove after confirming it works
try:
    from data_utils import fetch_fixtures, fetch_historical, get_real_time_stats, process_data
    st.success("Imported functions from data_utils.py successfully")
except ImportError as e:
    st.error(f"Failed to import from data_utils: {str(e)}")
    st.info("Make sure the file is named exactly 'data_utils.py' (lowercase + underscore) and is in the repo root.")

from model import PredictionEnsemble

st.set_page_config(page_title="UltimaPredict FC", layout="wide", page_icon="⚽")

st.title("UltimaPredict FC: The Greatest Football Predictions Platform")
st.markdown("Real-time AI predictions with ensemble ML + real APIs. Nairobi time: Jan 21, 2026.")

# ───────────────────────────────────────────────
# Load data & model
if 'df' not in st.session_state or 'model' not in st.session_state:
    with st.spinner("Loading historical data and training model..."):
        raw_df = fetch_historical()
        processed_df = process_data(raw_df)
        st.session_state.df = processed_df

        if processed_df.empty:
            st.error("No usable historical data loaded.")
            st.info("""
            Upload your E0.csv files (from football-data.co.uk) into the 'historical_data/' folder in this GitHub repo.
            After upload, wait 1–2 minutes for redeploy.
            """)
        else:
            st.success(f"Loaded {len(processed_df)} historical matches")

        try:
            st.session_state.model = PredictionEnsemble(st.session_state.df)
            st.success("Model trained successfully")
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")

# ───────────────────────────────────────────────
# League & match selector
league_options = [
    "Premier League (39)",
    "La Liga (140)",
    "Bundesliga (78)",
    "Serie A (135)",
    "Champions League (2)"
]
league = st.selectbox("Select League", league_options)
league_id = int(league.split("(")[1].strip(")"))

upcoming_df = fetch_fixtures(league_id=league_id, upcoming=True)

if not upcoming_df.empty:
    match_strs = upcoming_df['HomeTeam'] + " vs " + upcoming_df['AwayTeam'] + " (" + upcoming_df['Date'].dt.strftime('%Y-%m-%d') + ")"
    selected_match = st.selectbox("Select Upcoming Match", match_strs)
    parts = selected_match.split(" vs ")
    home = parts[0]
    away_part = parts[1]
    away = away_part.split(" (")[0]
    fixture_row = upcoming_df[upcoming_df['HomeTeam'] == home]
    fixture_id = fixture_row['fixture_id'].iloc[0] if not fixture_row.empty else None
else:
    st.warning("No upcoming matches found via API – using manual input")
    home = st.text_input("Home Team", "Arsenal")
    away = st.text_input("Away Team", "Man City")
    fixture_id = None

scenario = st.selectbox("Scenario Simulation", [None, "Injury to Home Key Player", "Injury to Away Key Player"])

# ───────────────────────────────────────────────
if st.button("Generate Prediction", type="primary"):
    if 'model' not in st.session_state:
        st.error("Model not loaded yet. Wait for data loading to finish.")
    else:
        with st.spinner("Running prediction..."):
            pred = st.session_state.model.predict(home, away, fixture_id, scenarios=scenario)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction")
            st.write(f"**Most likely outcome:** {pred['predicted_outcome']} ({pred['confidence']:.1f}% conf)")
            st.write(f"**Predicted score:** {home} **{pred['predicted_score']}** {away}")
            st.write(f"**Over 2.5 goals prob:** {pred['over_25_prob']:.1f}%")

            probs = pred['outcome_prob']
            st.write(f"{home} Win: {probs['Home Win']:.1f}% | Draw: {probs['Draw']:.1f}% | {away} Win: {probs['Away Win']:.1f}%")

        with col2:
            st.subheader("Betting Value")
            odds = pred.get('odds', {})
            st.write(f"Avg odds: {home} {odds.get('home_win', 'N/A'):.2f} | Draw {odds.get('draw', 'N/A'):.2f} | {away} {odds.get('away_win', 'N/A'):.2f}")
            st.write(f"Over 2.5 odds: {odds.get('over_2_5', 'N/A'):.2f}")

            evs = pred.get('ev', {})
            st.write(f"EV: {home} {evs.get('Home Win', 0):.2f} | Draw {evs.get('Draw', 0):.2f} | {away} {evs.get('Away Win', 0):.2f} | Over 2.5 {evs.get('Over 2.5', 0):.2f}")

            if pred.get('best_bet'):
                st.success(f"**Recommended bet:** {pred['best_bet']} (EV {evs[pred['best_bet']]:.2f})")

        # Probability bar chart
        fig, ax = plt.subplots()
        ax.bar(probs.keys(), probs.values(), color=['#4CAF50', '#FFC107', '#F44336'])
        ax.set_title("Outcome Probabilities")
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        st.subheader("Current Team Stats (API)")
        st.write(f"{home}: {get_real_time_stats(home)}")
        st.write(f"{away}: {get_real_time_stats(away)}")

# ───────────────────────────────────────────────
if st.button("Send this Prediction to Telegram"):
    if 'model' in st.session_state:
        bot = Bot(token=st.secrets.get("TELEGRAM_TOKEN", ""))
        message = (
            f"UltimaPredict FC\n"
            f"{home} vs {away}\n"
            f"→ {pred['predicted_outcome']} ({pred['confidence']:.1f}%)\n"
            f"Score: {pred['predicted_score']}\n"
            f"Over 2.5: {pred['over_25_prob']:.1f}%\n"
            f"Best bet: {pred.get('best_bet', 'None')} (EV {pred['ev'].get(pred.get('best_bet'), 0):.2f})"
        )
        async def send_msg():
            try:
                await bot.send_message(chat_id=st.secrets.get("TELEGRAM_CHAT_ID", ""), text=message)
                st.success("Sent to Telegram!")
            except Exception as e:
                st.error(f"Telegram send failed: {e}")

        asyncio.run(send_msg())
    else:
        st.warning("No prediction available yet.")

st.info("For entertainment & analysis only. Gamble responsibly.")
