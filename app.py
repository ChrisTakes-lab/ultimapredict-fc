# app.py — The interface to the greatest machine

import streamlit as st
from model import get_engine  # only imports the function, safe

st.set_page_config(page_title="The Greatest Machine", layout="wide")

st.title("The Greatest Prediction Machine Ever Built ⚡")
st.markdown("Predicting outcomes with unmatched precision. Odds. Stats. Value. Domination.")

engine = get_engine()  # lazy init on first use

# Demo / input section
col1, col2 = st.columns(2)
with col1:
    team_home = st.text_input("Home Team", "Man Utd")
with col2:
    team_away = st.text_input("Away Team", "Liverpool")

if st.button("Generate Prediction 🔥", type="primary"):
    with st.spinner("Consulting the oracle..."):
        result = engine.predict_with_context(team_home=team_home, team_away=team_away)
    
    st.success(f"Home Win Probability: **{result['proba_home_win']:.1%}**")
    st.metric("Value Bet on Home?", "YES — Smash It!" if result['value_bet_home'] else "No Edge")
    
    st.json(result)  # full output

# Sidebar for examples / debug
with st.sidebar:
    st.header("Sample Data")
    examples = engine._get_data_utils()[0]()  # get_examples
    st.dataframe(examples.head())
