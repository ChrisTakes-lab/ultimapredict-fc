import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
from torch import nn, optim
from scipy.stats import poisson, beta
import streamlit as st  # for warnings

from data_utils import get_external_prediction, get_odds, get_real_time_stats

class Net(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 3)  # Home Win, Draw, Away Win

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.softmax(self.fc3(x), dim=1)

class PredictionEnsemble:
    def __init__(self, df):
        self.df = df
        if df.empty or 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
            st.warning("Insufficient data for model training – using fallback mode")
            self.team_idx = {}
            self.xgb = self.rf = None
            self.net = None
            return

        self.team_idx = {t: i for i, t in enumerate(pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']])))}
        self.train_models()

    def train_models(self):
        # Basic features (expand with more from real_time_stats later)
        X = np.array([[self.team_idx.get(row['HomeTeam'], 0), self.team_idx.get(row['AwayTeam'], 0)]
                      for _, row in self.df.iterrows()])
        y = self.df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).fillna(1).astype(int)

        if len(X) < 10:
            st.warning("Too few matches for proper training – using simple fallback")
            self.xgb = self.rf = None
            self.net = Net(2)
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # XGBoost
        self.xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, eval_metric='mlogloss', random_state=42)
        self.xgb.fit(X_train, y_train)

        # Random Forest
        self.rf = RandomForestClassifier(n_estimators=200, random_state=42)
        self.rf.fit(X_train, y_train)

        # PyTorch NN
        self.net = Net(X.shape[1])
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(80):  # More epochs for better convergence
            inputs = torch.tensor(X_train, dtype=torch.float32)
            labels = torch.tensor(y_train.values, dtype=torch.long)
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Quick backtest
        if len(X_test) > 0:
            ensemble_preds = np.argmax((self.xgb.predict_proba(X_test) + self.rf.predict_proba(X_test) + self.net(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()) / 3, axis=1)
            acc = np.mean(ensemble_preds == y_test)
            st.write(f"Ensemble backtest accuracy: {acc:.2%} on {len(X_test)} samples")

    def poisson_goals(self, home, away):
        stats_h = get_real_time_stats(home)
        stats_a = get_real_time_stats(away)
        avg_goals = (self.df['FTHG'].mean() + self.df['FTAG'].mean()) / 2 if not self.df.empty else 2.7
        lambda_home = (stats_h.get('gf', 1.4) / max(stats_a.get('ga', 1), 0.5)) * avg_goals * 1.12  # Home advantage
        lambda_away = (stats_a.get('gf', 1.4) / max(stats_h.get('ga', 1), 0.5)) * avg_goals * 0.88

        # Dixon-Coles low-score adjustment
        rho = 0.05  # correlation parameter
        lambda_home_adj = lambda_home * (1 - rho * (lambda_home - 1) * (lambda_away - 1))
        lambda_away_adj = lambda_away * (1 - rho * (lambda_home - 1) * (lambda_away - 1))
        return max(lambda_home_adj, 0.3), max(lambda_away_adj, 0.3)

    def predict(self, home, away, fixture_id=None, scenarios=None):
        if self.xgb is None:
            return {
                'predicted_outcome': 'Draw',
                'predicted_score': '1-1',
                'over_25_prob': 50.0,
                'confidence': 50.0,
                'outcome_prob': {'Home Win': 33, 'Draw': 34, 'Away Win': 33},
                'ev': {'Over 2.5': 0.0},
                'best_bet': None,
                'odds': {}
            }

        idx_h = self.team_idx.get(home, 0)
        idx_a = self.team_idx.get(away, 0)
        x = np.array([[idx_h, idx_a]])
        x_t = torch.tensor(x, dtype=torch.float32)

        xgb_p = self.xgb.predict_proba(x)[0]
        rf_p = self.rf.predict_proba(x)[0]
        nn_p = self.net(x_t).detach().numpy()[0]
        our_prob = (xgb_p + rf_p + nn_p) / 3

        ext = get_external_prediction(fixture_id) if fixture_id else {'win_prob': {'Home Win': 33, 'Draw': 33, 'Away Win': 34}, 'over_25': 50}
        fused_prob = {}
        for k, i in zip(['Home Win', 'Draw', 'Away Win'], range(3)):
            fused_prob[k] = (our_prob[i] * 100 + ext['win_prob'].get(k, 33)) / 2

        lambda_h, lambda_a = self.poisson_goals(home, away)
        if scenarios == 'Injury to Home Key Player':
            lambda_h *= 0.75
        elif scenarios == 'Injury to Away Key Player':
            lambda_a *= 0.75
        score_home = round(lambda_h)
        score_away = round(lambda_a)
        score = f"{score_home}-{score_away}"
        total_goals = lambda_h + lambda_a
        over_25 = (1 - poisson.cdf(2, total_goals)) * 100
        fused_over = (over_25 + ext['over_25']) / 2

        max_p = max(our_prob)
        conf = beta.mean(1 + max_p * 100, 1 + (1 - max_p) * 100) * 100

        odds = get_odds(home, away)
        ev = {}
        for outcome, idx in [('Home Win', 0), ('Draw', 1), ('Away Win', 2)]:
            prob = fused_prob[outcome] / 100
            odd_key = outcome.lower().replace(' ', '_') + '_win'
            dec_odd = odds.get(odd_key, 2.0)
            ev[outcome] = (prob * dec_odd) - 1
        ev['Over 2.5'] = (fused_over / 100 * odds.get('over_2_5', 1.9)) - 1

        best_bet = max(ev, key=ev.get) if max(ev.values()) > 0.05 else None  # Threshold for +EV

        return {
            'outcome_prob': fused_prob,
            'predicted_outcome': max(fused_prob, key=fused_prob.get),
            'predicted_score': score,
            'over_25_prob': fused_over,
            'confidence': conf,
            'ev': ev,
            'best_bet': best_bet,
            'odds': odds
        }
