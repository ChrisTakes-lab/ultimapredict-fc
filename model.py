# model.py
# The core Prediction Engine — imports data_utils ONLY inside methods

import joblib
import numpy as np
import pandas as pd
# Your ML libs: from sklearn..., import xgboost as xgb, etc.

class PredictionEngine:
    def __init__(self, model_path='models/final_model.pkl'):
        """Initialize the greatest prediction brain"""
        self.model = joblib.load(model_path)  # or self.model = xgb.Booster(...), etc.
        self.is_loaded = True
        print("PredictionEngine initialized — ready to dominate")

    def _get_data_utils(self):
        """Lazy access to data functions — breaks any potential cycle"""
        from data_utils import get_examples, get_odds, get_real_time_stats
        return get_examples, get_odds, get_real_time_stats

    def predict(self, features_df):
        """Core prediction method — input: preprocessed features"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        probas = self.model.predict_proba(features_df)  # or .predict() for regression
        return probas[:, 1]  # e.g. prob home win, adjust per your target

    def predict_with_context(self, match_id=None, team_home=None, team_away=None):
        """Full powerful prediction with live data integration"""
        get_examples, get_odds, get_real_time_stats = self._get_data_utils()

        # Fetch real-time / context
        odds = get_odds(match_id)
        live_stats = get_real_time_stats(team_home, team_away)

        # Build feature vector (your real feature engineering here)
        features = pd.DataFrame({
            'home_odds': [odds['home']],
            'possession_home': [live_stats['possession_home']],
            # ... add all your engineered features
        })

        proba_home = self.predict(features)[0]
        
        return {
            'proba_home_win': proba_home,
            'proba_draw': 1 - proba_home - 0.15,  # placeholder; your real calc
            'proba_away_win': 0.15,               # placeholder
            'value_bet_home': proba_home > 1/odds['home'] + 0.05,  # simple value check
            'raw_odds': odds
        }

    def explain_prediction(self, features):
        """SHAP / LIME / custom explainability for the greatest transparency"""
        # Implement later — e.g. shap_values = self.explainer.shap_values(features)
        return "Explanation coming in v2 — trust but verify"

# Optional: singleton pattern if you want one global engine
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PredictionEngine()
    return _engine_instance
