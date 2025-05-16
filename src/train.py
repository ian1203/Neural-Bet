import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import os
import pandas as pd

from src.model import CornerLSTM
from src.match_dataset import MatchDataset

# Hyperparameters
BATCH_SIZE = 24
EPOCHS = 40
SEQUENCE_LENGTH = 7
LEARNING_RATE = 0.001
PATIENCE = 10

train_seasons = [
    "2009-10", "2010-11", "2011-12", "2012-13", "2013-14",
    "2014-15", "2015-16", "2016-17"
]
val_seasons = ["2017-18"]
test_seasons = ["2018-19"]

# Load dataset as DataFrame for manual balancing
csv_path = "/Users/ianvicente/Desktop/NeuralBet/Data/processed/2000-2022_with_form.csv"
df = pd.read_csv(csv_path)
train_df = df[df["Season"].isin(train_seasons) & (df["has_odds"] == True)].copy()

# Balance: undersample OVER games to match number of UNDERs
train_df["total_corners"] = train_df[["HC", "AC"]].sum(axis=1)
under_df = train_df[train_df["total_corners"] <= 9.5]
over_df = train_df[train_df["total_corners"] > 9.5].sample(len(under_df), random_state=42)
balanced_df = pd.concat([under_df, over_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save temporarily to new CSV
balanced_csv = "/Users/ianvicente/Desktop/NeuralBet/Data/processed/balanced_train_temp.csv"
balanced_df.to_csv(balanced_csv, index=False)

# Datasets and loaders
train_dataset = MatchDataset(balanced_csv, sequence_length=SEQUENCE_LENGTH, use_only_with_odds=False)
val_dataset = MatchDataset(csv_path, sequence_length=SEQUENCE_LENGTH, seasons=val_seasons, use_only_with_odds=True)
test_dataset = MatchDataset(csv_path, sequence_length=SEQUENCE_LENGTH, seasons=test_seasons, use_only_with_odds=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
INPUT_SIZE = train_dataset.X.shape[1]
model = CornerLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, dropout=0.2)
device = torch.device("cpu")
model.to(device)

# Loss with conditional quadratic penalty for UNDER games
mae_fn = nn.L1Loss()
mse_fn = nn.MSELoss()

def loss_fn(pred, target):
    base_loss = 0.7 * mae_fn(pred, target) + 0.3 * mse_fn(pred, target)
    with torch.no_grad():
        total_pred = pred.sum(dim=1)
        total_true = target.sum(dim=1)
        under_mask = (total_true <= 9.5)
        large_error_mask = torch.abs(total_pred - total_true) > 4
        penalty_mask = under_mask & large_error_mask
        penalty = ((total_pred - total_true) ** 2)[penalty_mask].mean() if penalty_mask.any() else 0.0
    return base_loss + 0.5 * penalty

# Early stopping setup
best_val_mae = float('inf')
best_epoch = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_preds, train_targets = [], []
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        train_preds.extend(preds.detach().cpu().numpy())
        train_targets.extend(y_batch.detach().cpu().numpy())

    train_preds = np.array(train_preds)
    train_targets = np.array(train_targets)
    train_mae = np.mean(np.abs(train_preds - train_targets))
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    train_over = (train_preds.sum(axis=1) > 9.5).astype(int)
    train_true = (train_targets.sum(axis=1) > 9.5).astype(int)
    train_hit = accuracy_score(train_true, train_over)
    train_prob = train_over.mean()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_mae = np.mean(np.abs(val_preds - val_targets))
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_over = (val_preds.sum(axis=1) > 9.5).astype(int)
    val_true = (val_targets.sum(axis=1) > 9.5).astype(int)
    val_hit = accuracy_score(val_true, val_over)
    val_prob = val_over.mean()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"[TRAIN] MAE: {train_mae:.3f} | RMSE: {train_rmse:.3f} | P(>9.5): {train_prob:.2f} | Hit Rate: {train_hit:.2f}")
    print(f"[ VAL ] MAE: {val_mae:.3f} | RMSE: {val_rmse:.3f} | P(>9.5): {val_prob:.2f} | Hit Rate: {val_hit:.2f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_epoch = epoch
        torch.save(model.state_dict(), "model/corner_lstm.pth")
        print("✅ Model saved.")
    elif epoch - best_epoch >= PATIENCE:
        print("🛑 Early stopping triggered.")
        break

# Load best model
model.load_state_dict(torch.load("model/corner_lstm.pth"))
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        test_preds.extend(pred.cpu().numpy())
        test_targets.extend(y_batch.cpu().numpy())

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_mae = np.mean(np.abs(test_preds - test_targets))
test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
test_over = (test_preds.sum(axis=1) > 9.5).astype(int)
test_true = (test_targets.sum(axis=1) > 9.5).astype(int)
test_hit = accuracy_score(test_true, test_over)
test_prob = test_over.mean()

print("\n[ TEST ] Results")
print(f"MAE: {test_mae:.3f} | RMSE: {test_rmse:.3f} | P(>9.5): {test_prob:.2f} | Hit Rate: {test_hit:.2f}")

# Export test predictions
df_out = pd.DataFrame({
    "home_corners_pred": test_preds[:, 0],
    "away_corners_pred": test_preds[:, 1],
    "home_corners_true": test_targets[:, 0],
    "away_corners_true": test_targets[:, 1],
    "total_pred": test_preds.sum(axis=1),
    "total_true": test_targets.sum(axis=1),
    "over_9_5_pred": (test_preds.sum(axis=1) > 9.5).astype(int),
    "over_9_5_true": (test_targets.sum(axis=1) > 9.5).astype(int),
    "abs_error": np.abs(test_preds.sum(axis=1) - test_targets.sum(axis=1))
})

os.makedirs("model", exist_ok=True)
df_out.to_csv("model/test_predictions_over9.5.csv", index=False)
print("📄 Test predictions saved to: model/test_predictions_over9.5.csv")
