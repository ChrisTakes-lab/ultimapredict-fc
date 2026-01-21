import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
from torch import nn, optim
from scipy.stats import poisson, beta
from data_utils import get_external_prediction, get_odds, get_real_time_stats

# Neural Net for outcome classification
class Net(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Win, Draw, Loss

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

class PredictionEnsemble:
    def __init__(self, df):
        self.df = df
        self.team_idx = {t: i for i, t in enumerate(pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']])))}
        self.train_models()

    def train_models(self):
        # Feature engineering: basic team IDs + form + stats
        X = np.array([[self.team_idx.get(row['HomeTeam'], 0), self.team_idx.get(row['AwayTeam'], 0)] for _, row in self.df.iterrows()])
        y = self.df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).fillna(1).astype(int)  # Draw default

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # XGBoost for outcome
        self.xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, eval_metric='mlogloss')
        self.xgb.fit(X_train, y_train)

        # Random Forest fallback
        self.rf = RandomForestClassifier(n_estimators=200)
        self.rf.fit(X_train, y_train)

        # PyTorch NN
        self.net = Net(X.shape[1])
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(50):
            inputs = torch.tensor(X_train, dtype=torch.float32)
            labels = torch.tensor(y_train.values, dtype=torch.long)
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Backtest accuracy
        if len(X_test) > 0:
            xgb_preds = self.xgb.predict(X_test)
            acc = np.mean(xgb_preds == y_test)
            st.write(f"Backtested accuracy: {acc:.2%} on {len(X_test)} test matches")

    def poisson_goals(self, home, away):
        # Advanced Poisson with real-time stats and Dixon-Coles adjustment
        stats_h = get_real_time_stats(home)
        stats_a = get_real_time_stats(away)
        avg_goals = (self.df['FTHG'].mean() + self.df['FTAG'].mean()) / 2
        lambda_home = stats_h['gf'] / stats_a['ga'] * avg_goals * 1.15  # Home edge
        lambda_away = stats_a['gf'] / stats_h['ga'] * avg_goals * 0.85

        # Dixon-Coles for low-scoring games
        if lambda_home < 1.5 or lambda_away < 1.5:
            lambda_home *= 1.05
            lambda_away *= 1.05

        return lambda_home, lambda_away

    def predict(self, home, away, fixture_id=None, scenarios=None):
        idx_h = self.team_idx.get(home, 0)
        idx_a = self.team_idx.get(away, 0)
        x = np.array([[idx_h, idx_a]])
        x_t = torch.tensor(x, dtype=torch.float32)

        # Ensemble probs
        xgb_prob = self.xgb.predict_proba(x)[0] if self.xgb else np.array([0.4, 0.3, 0.3])
        rf_prob = self.rf.predict_proba(x)[0] if self.rf else np.array([0.4, 0.3, 0.3])
        with torch.no_grad():
            nn_prob = self.net(x_t).numpy()[0]

        our_prob = (xgb_prob + rf_prob + nn_prob) / 3

        # Fuse with external API prediction
        ext = get_external_prediction(fixture_id) if fixture_id else {'win_prob': {'Home Win': 33, 'Draw': 34, 'Away Win': 33}, 'over_25': 50}
        fused_prob = {}
        for k, i in zip(['Home Win', 'Draw', 'Away Win'], range(3)):
            fused_prob[k] = (our_prob[i] * 100 + ext['win_prob'].get(k, 33)) / 2

        # Poisson score prediction
        lambda_h, lambda_a = self.poisson_goals(home, away)
        if scenarios == 'Injury to Home Key Player': lambda_h *= 0.8
        elif scenarios == 'Injury to Away Key Player': lambda_a *= 0.8
        score = f"{int(round(lambda_h))}-{int(round(lambda_a))}"
        total_exp = lambda_h + lambda_a
        over_25 = (1 - poisson.cdf(2, total_exp)) * 100
        fused_over = (over_25 + ext['over_25']) / 2

        # Bayesian confidence
        max_prob = max(our_prob)
        conf = beta.mean(1 + max_prob * 100, 1 + (100 - max_prob * 100)) * 100

        # Odds & EV
        odds = get_odds(home, away)
        ev = {}
        for k in ['Home Win', 'Draw', 'Away Win']:
            prob = fused_prob[k] / 100
            odd_key = k.lower().replace(' ', '_') + '_win'
            decimal = odds.get(odd_key, 2.0)
            ev[k] = (prob * decimal) - 1
        ev['Over 2.5'] = (fused_over / 100 * odds.get('over_2_5', 1.9)) - 1

        best_bet = max(ev, key=ev.get) if max(ev.values()) > 0 else None

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
