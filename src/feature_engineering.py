import pandas as pd

def compute_team_form(df, max_games=5):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    print("🕒 Missing dates:", df["Date"].isna().sum())

    form_features = ['form_pts', 'form_wins', 'form_goals_scored', 'form_goals_conceded',
                     'form_goal_diff', 'form_corners', 'form_shots', 'form_shots_on_target']

    for prefix in ['home_form_', 'away_form_']:
        for feat in form_features:
            df[f"{prefix}{feat}"] = None

    def get_team_form(data, team, date):
        past_games = data[((data["HomeTeam"] == team) | (data["AwayTeam"] == team)) &
                          (data["Date"] < date)].sort_values("Date", ascending=False).head(max_games)

        stats = dict.fromkeys(form_features, 0)

        for _, row in past_games.iterrows():
            if row["HomeTeam"] == team:
                gf, ga = row["FTHG"], row["FTAG"]
                cor, sht, sht_ont = row["HC"], row["HS"], row["HST"]
                result = row["FTR"]
            else:
                gf, ga = row["FTAG"], row["FTHG"]
                cor, sht, sht_ont = row["AC"], row["AS"], row["AST"]
                result = 'H' if row["FTR"] == 'A' else 'A' if row["FTR"] == 'H' else 'D'

            stats['form_goals_scored'] += gf
            stats['form_goals_conceded'] += ga
            stats['form_goal_diff'] += gf - ga
            stats['form_corners'] += cor
            stats['form_shots'] += sht
            stats['form_shots_on_target'] += sht_ont
            if result == 'H' and row["HomeTeam"] == team or result == 'A' and row["AwayTeam"] == team:
                stats['form_pts'] += 3
                stats['form_wins'] += 1
            elif result == 'D':
                stats['form_pts'] += 1

        return stats

    for i, row in df.iterrows():
        home_form = get_team_form(df.iloc[:i], row["HomeTeam"], row["Date"])
        away_form = get_team_form(df.iloc[:i], row["AwayTeam"], row["Date"])
        for feat in form_features:
            df.at[i, f"home_form_{feat}"] = home_form[feat]
            df.at[i, f"away_form_{feat}"] = away_form[feat]

    # Team strength features
    for side in ['HomeTeam', 'AwayTeam']:
        prefix = 'home' if side == 'HomeTeam' else 'away'
        team_stats = df.groupby([side, 'Season']).agg({
            'FTHG': 'mean',
            'FTAG': 'mean',
            'HC': 'mean',
            'AC': 'mean',
            'HS': 'mean',
            'AS': 'mean',
        }).reset_index()

        team_stats.columns = [side, 'Season',
                              f'{prefix}_avg_goals_for', f'{prefix}_avg_goals_against',
                              f'{prefix}_avg_corners_for', f'{prefix}_avg_corners_against',
                              f'{prefix}_avg_shots_for', f'{prefix}_avg_shots_against']

        df = df.merge(team_stats, on=[side, 'Season'], how='left')

    # Normalize Over 2.5 odds
    if "BbAv>2.5" in df.columns:
        df["Over25Odds"] = df["BbAv>2.5"]
    elif "B365>2.5" in df.columns:
        df["Over25Odds"] = df["B365>2.5"]
    elif "Avg>2.5" in df.columns:
        df["Over25Odds"] = df["Avg>2.5"]
    else:
        df["Over25Odds"] = None

    # Derived features from odds
    df["is_home_favorite"] = (df["B365H"] < df["B365A"]).astype(int)
    df["is_high_total_goals"] = ((df["Over25Odds"] < 2.0) & df["Over25Odds"].notna()).astype(int)
    df["odds_margin"] = df["B365A"] - df["B365H"]
    df["expected_goals_prob"] = (1 / df["Over25Odds"]).clip(upper=1.0).fillna(0)

    df["importance_score"] = (
        (df["is_high_total_goals"] * 1.5 + df["is_home_favorite"] * 1.0 + df["odds_margin"].abs())
    )

    # Derby flag
    derby_pairs = {
        ("Arsenal", "Tottenham"),
        ("Liverpool", "Everton"),
        ("Manchester United", "Manchester City"),
        ("Chelsea", "Arsenal"),
        ("Newcastle", "Sunderland"),
        ("Chelsea", "Manchester City"),
        ("Manchester City", "Tottenham"),
        ("Chelsea", "Tottenham"),
        ("Manchester City", "Arsenal")
    }
    df["is_derby"] = df.apply(lambda row: int((row["HomeTeam"], row["AwayTeam"]) in derby_pairs or
                                              (row["AwayTeam"], row["HomeTeam"]) in derby_pairs), axis=1)

    # Head-to-head stats
    def get_h2h_stats(data, home, away, date):
        past = data[((data["HomeTeam"] == home) & (data["AwayTeam"] == away)) |
                    ((data["HomeTeam"] == away) & (data["AwayTeam"] == home))]
        past = past[past["Date"] < date].sort_values("Date", ascending=False).head(5)
        return {
            "h2h_avg_corners": past[["HC", "AC"]].sum(axis=1).mean(),
            "h2h_avg_goals": past[["FTHG", "FTAG"]].sum(axis=1).mean(),
            "h2h_total_matches": len(past)
        }

    df["h2h_avg_corners"] = 0.0
    df["h2h_avg_goals"] = 0.0
    df["h2h_total_matches"] = 0

    for i, row in df.iterrows():
        stats = get_h2h_stats(df.iloc[:i], row["HomeTeam"], row["AwayTeam"], row["Date"])
        df.at[i, "h2h_avg_corners"] = stats["h2h_avg_corners"]
        df.at[i, "h2h_avg_goals"] = stats["h2h_avg_goals"]
        df.at[i, "h2h_total_matches"] = stats["h2h_total_matches"]

    # Additional features for UNDER prediction
    df["team_avg_total_corners"] = (df["home_form_form_corners"] + df["away_form_form_corners"]) / 2
    df["is_high_corners_match"] = ((df["HC"] + df["AC"]) > 11).astype(int)

    df["low_form_total_corners"] = (
        (df["home_form_form_corners"] + df["away_form_form_corners"]) < 20
    ).astype(int)

    df["low_attack_symmetry"] = (
        (df["home_avg_shots_for"] + df["away_avg_shots_for"]) < 20
    ).astype(int)

    df["low_conceding_pair"] = (
        (df["home_avg_goals_against"] + df["away_avg_goals_against"]) < 2.0
    ).astype(int)

    df["is_low_total_goals"] = (
        (df["Over25Odds"] > 2.4) & df["Over25Odds"].notna()
    ).astype(int)

    df.update(df.select_dtypes(include=['object']).apply(pd.to_numeric, errors='coerce'))

    return df
