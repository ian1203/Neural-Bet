import os
import pandas as pd

# Define base path
base_path = "/Users/ianvicente/Desktop/NeuralBet/Data/raw"


def load_multiple_seasons(data_dir, seasons):
    """
    Load and combine CSVs from multiple seasons into a single DataFrame.
    Adds a 'Season' column to each entry.
    """
    all_dfs = []
    for season in seasons:
        file_path = os.path.join(data_dir, f"{season}.csv")
        print(f"🔄 Loading: {file_path}")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["Season"] = season
            all_dfs.append(df)
        else:
            print(f"⚠️ File not found: {file_path}")
    return pd.concat(all_dfs, ignore_index=True)