import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MatchDataset(Dataset):
    def __init__(self, csv_path, sequence_length=1, seasons=None, use_only_with_odds=False):
        self.df = pd.read_csv(csv_path)

        if use_only_with_odds:
            self.df = self.df[self.df["has_odds"] == True]
        if seasons:
            self.df = self.df[self.df["Season"].isin(seasons)]
        self.df = self.df.reset_index(drop=True)
        print("📦 Final dataset size:", len(self.df))
        print("🗓️  Seasons in dataset:", self.df["Season"].unique())
        print("📊 Value counts for Season:\n", self.df["Season"].value_counts())

        self.features = [
            # Form stats
            'home_form_form_pts', 'home_form_form_wins', 'home_form_form_goals_scored',
            'home_form_form_goals_conceded', 'home_form_form_goal_diff',
            'home_form_form_corners', 'home_form_form_shots', 'home_form_form_shots_on_target',
            'away_form_form_pts', 'away_form_form_wins', 'away_form_form_goals_scored',
            'away_form_form_goals_conceded', 'away_form_form_goal_diff',
            'away_form_form_corners', 'away_form_form_shots', 'away_form_form_shots_on_target',

            # Odds-based & categorical features
            "is_home_favorite", "is_high_total_goals", "odds_margin", "expected_goals_prob",

            # Historical matchup & team strength
            'h2h_avg_corners', 'h2h_avg_goals', 'h2h_total_matches',
            'importance_score', 'is_derby',
            'home_avg_goals_for', 'home_avg_goals_against',
            'home_avg_corners_for', 'home_avg_corners_against',
            'home_avg_shots_for', 'home_avg_shots_against',
            'away_avg_goals_for', 'away_avg_goals_against',
            'away_avg_corners_for', 'away_avg_corners_against',
            'away_avg_shots_for', 'away_avg_shots_against',

            # UNDER-oriented features
            'team_avg_total_corners', 'low_form_total_corners', 'low_attack_symmetry',
            'low_conceding_pair', 'is_low_total_goals'
        ]

        self.target_columns = ['HC', 'AC']
        self.sequence_length = sequence_length

        self.df[self.features] = self.df[self.features].fillna(0)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.df[self.features])
        self.y = self.df[self.target_columns].values

        print(f"✅ Input shape: {self.X.shape} | Target shape: {self.y.shape}")

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)
