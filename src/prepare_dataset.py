from src.data_loader import load_multiple_seasons
from src.feature_engineering import compute_team_form

if __name__ == "__main__":
    # Step 1: Load data
    df = load_multiple_seasons(
        "/Users/ianvicente/Desktop/NeuralBet/Data/raw",
        [
            "2000-01", "2001-02", "2002-03", "2003-04", "2004-05", "2005-06",
            "2006-07", "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
            "2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
            "2018-19", "2019-20", "2020-21", "2021-22"
        ]
    )

    # Normalize odds column to a single name for Over 2.5 goals
    if "BbAv>2.5" in df.columns:
        df["Over25Odds"] = df["BbAv>2.5"]
    elif "B365>2.5" in df.columns:
        df["Over25Odds"] = df["B365>2.5"]
    elif "Avg>2.5" in df.columns:
        df["Over25Odds"] = df["Avg>2.5"]
    else:
        df["Over25Odds"] = None  # fallback if missing

    # Step 2: Add form features
    df = compute_team_form(df)

    df["has_odds"] = df["B365H"].notna() & df["B365A"].notna() & df["Over25Odds"].notna()


    # Optional print to verify balance
    print("✅ Rows with odds:", df["has_odds"].sum())
    print("❌ Rows without odds:", (~df["has_odds"]).sum())

    # Step 3: Save to processed folder
    df.to_csv("/Users/ianvicente/Desktop/NeuralBet/Data/processed/2000-2022_with_form.csv", index=False)
    print("✅ Processed dataset saved to: data/processed/2000-2022_with_form.csv")