# Libraries
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import pathlib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import random
from itertools import product

# Paths
THIS_FILE = pathlib.Path(__file__)
PACKAGE_ROOT = THIS_FILE.parent.parent
TRAINING_DATA = PACKAGE_ROOT / "data" / "training.csv"
VALIDATION_DATA = PACKAGE_ROOT / "data" / "validation.csv"
TESTING_DATA = PACKAGE_ROOT / "data" / "testing.csv"

# Static Parameters
N_DAYS = 10
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
PATIENCE = 5
DEVICE = torch.device(
    "cpu"
)  # I will force PyTorch to use only CPU to make sure the code is reproduceable in all machines

# Setting seeds for as much reproducibility as possible
"""
From pytorch.org:
Completely reproducible results are not guaranteed across PyTorch releases, individual commits, 
or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, 
even when using identical seeds.
"""
SEED = 2187
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Features
FEATURES = ["Open", "High", "Low", "Close", "Volume", "Daily_Return"]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Helper Functions
def prepare_sequences(
    df: pd.DataFrame,
    n_days: int = N_DAYS,
    feature_columns: list = FEATURES,
) -> tuple:
    """
    Convert stock price data into sequences of N_DAYS length for use in a CNN.
    Returns X (features) and y (labels) as numpy arrays.
    """
    X, y = [], []

    for i in range(len(df) - n_days):
        X.append(
            df.iloc[i : i + n_days][feature_columns].values.T
        )  # (features, time-steps)
        y.append(df.iloc[i + n_days]["Extreme_Event"])

    X = np.array(X, dtype=np.float32)  # Shape: (samples, features, time-steps)
    y = np.array(y, dtype=np.int64)  # Labels: Binary classification

    return X, y


def prepare_data(
    df: pd.DataFrame,
    n_batch: int,
    shuffle: bool = True,
) -> tuple:
    """
    Load training, validation, and testing data and convert them into DataLoader objects.
    """
    logging.info("Loading and preparing data")

    # Convert data into CNN-ready format
    X, y = prepare_sequences(df)

    # Convert NumPy arrays to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Create DataLoader
    generator = torch.Generator().manual_seed(SEED)
    processed_data = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=n_batch,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )

    logging.info(
        f"Loaded {len(X_tensor)} samples. Batch size: {n_batch}. Shuffling: {shuffle}"
    )

    return processed_data


def scale_train_val(
    train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list
) -> tuple:
    """
    Fit a StandardScaler on the training set, then transform both train and val.
    """
    logging.info("Scaling the data using StandardScaler")
    scaler = StandardScaler()

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()

    train_scaled[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    val_scaled[feature_columns] = scaler.transform(val_df[feature_columns])

    logging.info("Data scaling completed successfully.")
    return train_scaled, val_scaled, scaler


def report_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> tuple:
    """
    Compute confusion matrix, accuracy, precision, recall, and F1 Score.
    """
    # Computing
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100

    logging.info(f"\n{model_name} Confusion Matrix:\n{cm}")
    logging.info(f"{model_name} Accuracy: {accuracy:.2f}%")
    logging.info(f"{model_name} Precision: {precision:.2f}%")
    logging.info(f"{model_name} Recall: {recall:.2f}%")
    logging.info(f"{model_name} F1 Score: {f1:.2f}%")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm.tolist(),
    }


# Simple Moving Averages (SMA)
def calculate_sma(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df["SMA"] = df["Close"].rolling(window=window).mean()
    return df


# Exponential Moving Averages (EMA)
def calculate_ema(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df["EMA"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


# Bollinger Bands
def calculate_bollinger_bands(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df["MA_Close"] = df["Close"].rolling(window=window).mean()
    df["STD_Close"] = df["Close"].rolling(window=window).std()
    df["Bollinger_Upper"] = df["MA_Close"] + (df["STD_Close"] * 2)
    df["Bollinger_Lower"] = df["MA_Close"] - (df["STD_Close"] * 2)
    df["Bollinger_Band_Width"] = df["Bollinger_Upper"] - df["Bollinger_Lower"]
    return df


# Average True Range (ATR)
def calculate_atr(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df["Prev_Close"] = df["Close"].shift(1)
    df["High_Low"] = df["High"] - df["Low"]
    df["High_PrevClose"] = abs(df["High"] - df["Prev_Close"])
    df["Low_PrevClose"] = abs(df["Low"] - df["Prev_Close"])
    df["True_Range"] = df[["High_Low", "High_PrevClose", "Low_PrevClose"]].max(axis=1)
    df["ATR"] = df["True_Range"].rolling(window=window).mean()
    df.drop(
        ["High_Low", "High_PrevClose", "Low_PrevClose", "True_Range", "Prev_Close"],
        axis=1,
        inplace=True,
    )
    return df


# Temporal Indicators
def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Weekday"] = df["Date"].dt.weekday
    df["Is_Weekend"] = df["Weekday"].isin([5, 6]).astype(int)
    df["Is_Monday"] = (df["Weekday"] == 0).astype(int)
    df["Is_Month_Start"] = df["Date"].dt.is_month_start.astype(int)
    df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)
    return df


# Volume-Weighted Average Price (VWAP)
def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    cumulative_volume_price = (df["Close"] * df["Volume"]).cumsum()
    cumulative_volume = df["Volume"].cumsum()
    df["VWAP"] = cumulative_volume_price / cumulative_volume
    return df


# Volume Spike (compared to a rolling average)
def calculate_volume_spike(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df["Rolling_Volume"] = df["Volume"].rolling(window=window).mean()
    df["Volume_Spike"] = (df["Volume"] - df["Rolling_Volume"]) / df["Rolling_Volume"]
    df.drop("Rolling_Volume", axis=1, inplace=True)
    return df


# Lag Features
def calculate_lag_features(df: pd.DataFrame, lags=[1, 2]) -> pd.DataFrame:
    for lag in lags:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Daily_Return_Lag_{lag}"] = df["Daily_Return"].shift(lag)
    return df


# Covid-19 and Apple Event Features (Adjusted)
def world_events(df: pd.DataFrame) -> pd.DataFrame:
    df["Covid"] = ((df["Date"] >= "2020-02-01") & (df["Date"] <= "2022-12-31")).astype(
        int
    )
    df["Apple_Event"] = (
        (df["Date"].dt.month == 9) & (df["Date"].dt.day.between(7, 20))
    ).astype(int)
    return df


# Rolling average of the standard deviation
def calculate_std_rolling(
    df: pd.DataFrame, feature_name: str, window: int = 5, suffix: str = None
) -> pd.DataFrame:
    # Create a unique column name if suffix is provided
    suffix = suffix or f"std_{window}"
    new_col = f"{suffix}_{feature_name}"

    # Calculate rolling standard deviation
    df[new_col] = df[feature_name].rolling(window=window).std()

    return df


def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX) Measures the strength of a trend (regardless of direction)
    """
    # Shift the 'Close' price to calculate previous close
    df["Prev_Close"] = df["Close"].shift(1)

    # Calculate True Range (TR)
    df["High_Low"] = df["High"] - df["Low"]
    df["High_PrevClose"] = abs(df["High"] - df["Prev_Close"])
    df["Low_PrevClose"] = abs(df["Low"] - df["Prev_Close"])
    df["TR"] = df[["High_Low", "High_PrevClose", "Low_PrevClose"]].max(axis=1)

    # Calculate Directional Movement
    df["+DM"] = np.where(
        (df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]),
        np.maximum(df["High"] - df["High"].shift(1), 0),
        0,
    )
    df["-DM"] = np.where(
        (df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)),
        np.maximum(df["Low"].shift(1) - df["Low"], 0),
        0,
    )

    # Smooth the indicators with a rolling mean
    df["TR_smooth"] = df["TR"].rolling(window=window).mean()
    df["+DI"] = 100 * (df["+DM"].rolling(window=window).mean() / df["TR_smooth"])
    df["-DI"] = 100 * (df["-DM"].rolling(window=window).mean() / df["TR_smooth"])

    # Calculate ADX
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100
    df["ADX"] = df["DX"].rolling(window=window).mean()

    # Drop intermediate columns to clean the DataFrame
    df.drop(
        [
            "Prev_Close",
            "High_Low",
            "High_PrevClose",
            "Low_PrevClose",
            "TR",
            "+DM",
            "-DM",
            "TR_smooth",
            "DX",
        ],
        axis=1,
        inplace=True,
    )

    return df


def calculate_rsi(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Relative Strength Index (RSI)
    Detects momentum
    """
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def calculate_roc(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Price Rate of Change (ROC)
    Measures the percentage change between the current price and the price a certain number of periods ago
    """
    df["ROC"] = (
        (df["Close"] - df["Close"].shift(window)) / df["Close"].shift(window)
    ) * 100
    return df


def calculate_trend_slope(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Captures the slope (rate of change) over a window, indicating the strength of the trend.
    """
    df["Trend_Slope"] = (
        df["Close"]
        .rolling(window=window)
        .apply(lambda x: np.polyfit(range(window), x, 1)[0], raw=True)
    )
    return df


def calculate_ema_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Captures the difference between consecutive EMAs
    to measure acceleration in price movement.
    """
    df["EMA_Diff"] = df["EMA"].diff()
    return df


# Main feature engineering function
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_bollinger_bands(df)
    df = calculate_atr(df)
    df = calculate_vwap(df)
    df = calculate_volume_spike(df)
    # df = temporal_features(df)
    df = calculate_lag_features(df)
    df = calculate_std_rolling(df, feature_name="Close")
    df = calculate_std_rolling(df, feature_name="Daily_Return")
    # df = calculate_adx(df)
    # df = calculate_rsi(df)
    # df = calculate_roc(df)
    # df = calculate_trend_slope(df)
    # df = calculate_ema_diff(df)
    # df = world_events(df)

    df.dropna(inplace=True)
    return df


# Model Class
class TemporalCNN(nn.Module):
    def __init__(
        self,
        num_features,
        sequence_length,
        num_filters,
        kernel_size,
        dilation,
        dropout_rate,
    ):
        super(TemporalCNN, self).__init__()

        def get_padding(k, d):
            return d * (k - 1) // 2

        self.conv1 = nn.Conv1d(
            num_features,
            num_filters,
            kernel_size,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            2 * num_filters,
            kernel_size,
            dilation=dilation * 2,
            padding=get_padding(kernel_size, dilation * 2),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * 2 * sequence_length, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train_and_evaluate_model(config: dict, df_train, df_val) -> tuple:
    """
    Function to train and evaluate a Temporal CNN model.
    """
    class_weights = torch.tensor([1.0, config["weight_minority"]]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    torch.manual_seed(SEED)
    model = TemporalCNN(
        len(FEATURES),
        N_DAYS,
        config["num_filters"],
        config["kernel_size"],
        config["dilation"],
        config["dropout_rate"],
    ).to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")
    patience_counter = 0
    min_delta = 0.001
    best_state_dict = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_samples = 0, 0

        for X_batch, y_batch in df_train:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimiser.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        train_loss = total_loss / total_samples

        # Inference on the validation set
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in df_val:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                batch_loss = criterion(logits, y_batch)
                val_loss += batch_loss.item() * X_batch.size(0)

                val_total += y_batch.size(0)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= val_total
        val_accuracy = 100.0 * val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, zero_division=0) * 100

        logging.info(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"- Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val Acc: {val_accuracy:.2f}% "
            f"- Val F1: {val_f1:.2f}%"
        )

        # Early stopping based on validation loss
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered.")
                break

    # Load best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Report the best metrics
    logging.info("Evaluating TCNN on validation set")
    # Let's predict the validation set
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in df_val:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = model(X_batch).argmax(dim=1)

            preds.extend(y_pred.numpy())
            labels.extend(y_batch.numpy())

    # Compute metrics
    val_metrics = report_metrics(labels, preds, "TCNN (Validation Set)")

    return val_metrics, model


#################### MAIN ####################


def main(
    evaluate: bool = False,
    resample: bool = False,
    save_results: bool = False,
    hyperparameter_tuning: bool = False,
    apply_feature_engineering: bool = False,
):

    # Load the csv data
    logging.info("Loading data")
    train_raw = pd.read_csv(TRAINING_DATA)
    val_raw = pd.read_csv(VALIDATION_DATA)

    # Original Features
    FEATURES = ["Open", "High", "Low", "Close", "Volume", "Daily_Return"]

    if apply_feature_engineering:
        # Feature Engineering
        logging.info("Feature Engineering")
        train_engineered = feature_engineering(train_raw)
        val_engineered = feature_engineering(val_raw)

        # Adding the new features to the list
        # Let's find all the columns not in the original FEATURES list Also not in ['Date', 'Extreme_Event', 'Adj Close']
        to_append = [
            col
            for col in train_engineered.columns
            if col not in FEATURES + ["Date", "Extreme_Event", "Adj Close"]
        ]
        FEATURES += to_append
        logging.info(f"Here is a list with all the features: {FEATURES}")
    else:
        logging.info("Proceeding without feature engineering")
        train_engineered = train_raw.copy()
        val_engineered = val_raw.copy()

    # Scaling the data
    # Neural Nets are sensitive to the scale of the input data. For random forest, we didn't need to rescale the data.
    logging.info("Scaling the data")
    train_scaled, val_scaled, scaler = scale_train_val(
        train_engineered, val_engineered, FEATURES
    )

    if resample:
        logging.info(
            f"Before Resampling: Extreme Events = "
            f"{train_scaled['Extreme_Event'].mean() * 100:.2f}% of train"
        )
        logging.info("Applying SMOTE resampling to balance the training dataset")
        smote = SMOTE(random_state=SEED)
        X_res, y_res = smote.fit_resample(
            train_scaled[FEATURES], train_scaled["Extreme_Event"]
        )
        train_scaled = pd.DataFrame(X_res, columns=FEATURES)
        train_scaled["Extreme_Event"] = y_res
        logging.info(
            f"After SMOTE, the extreme events are now "
            f"{train_scaled['Extreme_Event'].mean() * 100:.2f}% of train"
        )
    else:
        logging.info("Proceeding without resampling")

    # Prepare the data
    df_train = prepare_data(train_scaled, BATCH_SIZE, shuffle=True)
    df_val = prepare_data(val_scaled, BATCH_SIZE, shuffle=False)

    if hyperparameter_tuning:
        # Hyperparameter Grid
        param_grid = {
            "num_filters": [32, 64, 128],
            "kernel_size": [3, 5, 7],
            "dilation": [1, 2, 4, 8],
            "dropout_rate": [0.2, 0.3, 0.5, 0.6],
            "learning_rate": [0.01, 0.001, 0.0005, 0.0001],
            "weight_minority": [1, 2, 4.06, 5, 7, 10],
        }

        # Hyperparameter Tuning
        grid = list(product(*param_grid.values()))
        results_df = pd.DataFrame()
        config_df = pd.DataFrame()
        for params in grid:
            config = dict(zip(param_grid.keys(), params))
            logging.info(f"Training with config: {config}")
            result, _ = train_and_evaluate_model(config, df_train, df_val)
            results_df = pd.concat(
                [results_df, pd.DataFrame([result])], ignore_index=True
            )
            config_df = pd.concat(
                [config_df, pd.DataFrame([config])], ignore_index=True
            )

        all_results_df = pd.concat([config_df, results_df], axis=1)

        if save_results:
            # Save the results
            all_results_df.to_csv(
                PACKAGE_ROOT / "data" / "model_improvement_results_2_layers.csv",
                index=False,
            )

        # Let's define the best params and predict in the validation set
        best_params = all_results_df.sort_values(
            by=["F1 Score", "Recall"], ascending=False
        ).iloc[0]

        best_params = best_params.to_dict()
        logging.info(f"Best Parameters: {best_params}")
    else:
        logging.info("Proceeding with the best parameters found in the previous run")
        best_params = {
            "num_filters": 32,
            "kernel_size": 3,
            "dilation": 1,
            "dropout_rate": 0.6,
            "learning_rate": 0.001,
            "weight_minority": 4.06,
        }

    # Train and evaluate the model
    final_result, model = train_and_evaluate_model(best_params, df_train, df_val)

    if save_results:
        # Save the results
        final_result_df = pd.DataFrame([final_result])
        final_result_df.to_csv(
            PACKAGE_ROOT / "data" / "final_metrics_validation_dataset.csv", index=False
        )

    if evaluate:
        logging.info("Evaluating the model on the test set")
        logging.info("We will be using are:")
        logging.info(final_result)
        logging.info("Now, let's evaluate the model on the test set.")
        # Load the test data
        test_raw = pd.read_csv(TESTING_DATA)
        if apply_feature_engineering:
            logging.info("Feature Engineering on the test set")
            test_engineered = feature_engineering(test_raw)
            test_scaled = test_engineered.copy()
            test_scaled[FEATURES] = scaler.transform(test_engineered[FEATURES])
        else:
            logging.info("Proceeding without feature engineering on the test set")
            test_scaled = test_raw.copy()
            test_scaled[FEATURES] = scaler.transform(test_raw[FEATURES])

        # Prepare the data
        df_test = prepare_data(test_scaled, BATCH_SIZE, shuffle=False)

        # Predicting the test set
        # Inference on test
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in df_test:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        final_test_metrics = report_metrics(
            all_labels, all_preds, "TCNN Final (Test Set)"
        )
        logging.info("Final test evaluation completed.")

        if save_results:
            # Save the results
            final_test_metrics_df = pd.DataFrame([final_test_metrics])
            final_test_metrics_df.to_csv(
                PACKAGE_ROOT / "data" / "final_test_metrics.csv", index=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improving the Temporal CNN.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model.")
    parser.add_argument(
        "--resample", action="store_true", help="Apply SMOTE resampling."
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results as CSV."
    )
    parser.add_argument(
        "--hyperparameter_tuning", action="store_true", help="Tune Hyperparameters."
    )
    parser.add_argument(
        "--apply_feature_engineering", action="store_true", help="Engineer Features."
    )
    args = parser.parse_args()

main(
    evaluate=args.evaluate,
    resample=args.resample,
    save_results=args.save_results,
    hyperparameter_tuning=args.hyperparameter_tuning,
    apply_feature_engineering=args.apply_feature_engineering,
)

#################### APPENDIX: EDA ####################
"""
# Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming that we have the best_params and the model from the previous run
# Let's predict the validation set and do some error analysis
config = best_params
class_weights = torch.tensor([1.0, config["weight_minority"]]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

torch.manual_seed(SEED)
model = TemporalCNN(len(FEATURES), N_DAYS, config['num_filters'], config['kernel_size'], config['dilation'], config['dropout_rate']).to(DEVICE)
optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])

best_val_loss = float("inf")
patience_counter = 0
min_delta = 0.001
best_state_dict = None

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_samples = 0, 0

    for X_batch, y_batch in df_train:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimiser.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

    train_loss = total_loss / total_samples

    # Inference on the validation set
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in df_val:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            batch_loss = criterion(logits, y_batch)
            val_loss += batch_loss.item() * X_batch.size(0)

            val_total += y_batch.size(0)
            val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    val_loss /= val_total
    val_accuracy = 100.0 * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, zero_division=0) * 100

    logging.info(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"- Train Loss: {train_loss:.4f} "
        f"- Val Loss: {val_loss:.4f} "
        f"- Val Acc: {val_accuracy:.2f}% "
        f"- Val F1: {val_f1:.2f}%"
    )

    # Early stopping based on validation loss
    if val_loss < (best_val_loss - min_delta):
        best_val_loss = val_loss
        best_state_dict = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            logging.info("Early stopping triggered.")
            break

# Load best checkpoint
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# Report the best metrics
logging.info("Evaluating TCNN on validation set")
# Let's predict the validation set
model.eval()
preds, labels = [], []

with torch.no_grad():
    for X_batch, y_batch in df_val:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        y_pred = model(X_batch).argmax(dim=1)

        preds.extend(y_pred.numpy())
        labels.extend(y_batch.numpy())

# Concatenating val_raw and preds columnwise (first 10 rows of val_raw should be omitted)
val_raw_for_eda = val_raw.iloc[10:].copy()
val_raw_for_eda["Extreme_Event_Pred"] = preds
val_raw_for_eda["Correct_Pred_Flag"] = val_raw_for_eda["Extreme_Event"] == val_raw_for_eda["Extreme_Event_Pred"]
correct_preds = val_raw_for_eda[(val_raw_for_eda['Extreme_Event'] == 1) & (val_raw_for_eda['Correct_Pred_Flag'] == 1)]
incorrect_preds = val_raw_for_eda[(val_raw_for_eda['Extreme_Event'] == 1) & (val_raw_for_eda['Correct_Pred_Flag'] == 0)]

# Boxplots for the correct and incorrect predictions
df_combined = pd.concat([
        correct_preds.assign(Prediction_Result='Correct'),
        incorrect_preds.assign(Prediction_Result='Incorrect')
    ])

with plt.style.context('seaborn-v0_8-whitegrid'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, feature in enumerate(FEATURES):
        sns.boxplot(x='Prediction_Result', y=feature, data=df_combined,
                    hue='Prediction_Result', palette='Set2', ax=axes[idx], legend=False)
        
        axes[idx].set_title(f'{feature} by Prediction Result')
        axes[idx].grid(True, linestyle='--', alpha=0.5)
        axes[idx].set_xlabel('')  # Optional: remove x-label redundancy
        axes[idx].set_ylabel(feature)
    plt.suptitle('Feature Distributions for Correct vs Incorrect Predictions (Extreme Events only)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Let's see also the performance per day of the week
val_raw_for_eda["Date"] = pd.to_datetime(val_raw_for_eda["Date"])
val_raw_for_eda["Day_of_Week"] = val_raw_for_eda["Date"].dt.day_name()
weekday_accuracy = val_raw_for_eda.groupby('Day_of_Week')['Correct_Pred_Flag'].mean()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_accuracy = weekday_accuracy.reindex(weekday_order)

# Plotting the f1 scores by day of the week
weekday_f1_scores = []

for day in weekday_order:
    daily_data = val_raw_for_eda[val_raw_for_eda['Day_of_Week'] == day]
    if not daily_data.empty:
        f1 = f1_score(daily_data['Extreme_Event'], daily_data['Extreme_Event_Pred'])
        weekday_f1_scores.append(f1)
    else:
        weekday_f1_scores.append(None)

with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(10, 5))
    plt.plot(weekday_order, weekday_f1_scores, marker='o', linewidth=2, color='purple')
    plt.title('F1-Score by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

# Let's see the trading volume by day of the week
weekday_volume = val_raw_for_eda.groupby('Day_of_Week')['Volume'].mean()
weekday_volume = weekday_volume.reindex(weekday_order)

with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(10, 5))
    plt.plot(weekday_order, weekday_volume, marker='o', linewidth=2, color='blue')
    plt.title('Average Trading Volume by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Trading Volume')
    plt.grid(True)
    plt.show()

## Now let's see all our mistakes in the validation dataset
correct_preds = val_raw_for_eda[(val_raw_for_eda['Extreme_Event'] == 1) & (val_raw_for_eda['Correct_Pred_Flag'] == 1)]
False_Negatives = val_raw_for_eda[(val_raw_for_eda['Extreme_Event'] == 0) & (val_raw_for_eda['Correct_Pred_Flag'] == 0)]
False_Positives = val_raw_for_eda[(val_raw_for_eda['Extreme_Event'] == 1) & (val_raw_for_eda['Correct_Pred_Flag'] == 0)]

with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(14, 7))

    # Plot Adjusted Close price
    plt.plot(val_raw_for_eda['Date'], val_raw_for_eda['Adj Close'], linewidth=1, label='Adj Close')

    # Plot correct and incorrect predictions with different markers/colors
    plt.scatter(correct_preds['Date'], correct_preds['Adj Close'], color='green', label='Correct Predictions')
    plt.scatter(False_Negatives['Date'], False_Negatives['Adj Close'], color='red', label='False Negatives')
    plt.scatter(False_Positives['Date'], False_Positives['Adj Close'], color='orange', label='False Positives')

    # Enhancing plot readability
    plt.title('Adjusted Close Price with Extreme Events', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ACF and PACF graphs
ts_adj_close = val_raw_for_eda.set_index('Date')['Adj Close']
with plt.style.context('seaborn-v0_8-whitegrid'):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    # Plot Autocorrelation Function (ACF)
    plot_acf(ts_adj_close, ax=ax[0], lags=40)
    ax[0].set_title('Autocorrelation (ACF) of Adj Close')

    # Plot Partial Autocorrelation Function (PACF)
    plot_pacf(ts_adj_close, ax=ax[1], lags=40, method='ywm')
    ax[1].set_title('Partial Autocorrelation (PACF) of Adj Close')

    plt.tight_layout()
    plt.show()

# Let's take first differences of the Adj Close and plot the ACF and PACF graphs
ts_adj_close_diff = ts_adj_close.diff().dropna()
with plt.style.context('seaborn-v0_8-whitegrid'):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    # Plot Autocorrelation Function (ACF)
    plot_acf(ts_adj_close_diff, ax=ax[0], lags=40)
    ax[0].set_title('Autocorrelation (ACF) of First Differences of Adj Close')

    # Plot Partial Autocorrelation Function (PACF)
    plot_pacf(ts_adj_close_diff, ax=ax[1], lags=40, method='ywm')
    ax[1].set_title('Partial Autocorrelation (PACF) of First Differences of Adj Close')

    plt.tight_layout()
    plt.show()
    
# Plotting ts_adj_close_diff time series data
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(14, 7))
    plt.plot(ts_adj_close_diff, linewidth=1, label='First Differences of Adj Close')
    plt.title('First Differences of Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('First Differences of Adj Close')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Let's log-transform the Adj Close and then take first differences
ts_adj_close_log_diff = np.log(ts_adj_close).diff().dropna()
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(14, 7))
    plt.plot(ts_adj_close_log_diff, linewidth=1, label='Log First Differences of Adj Close')
    plt.title('Log First Differences of Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Log First Differences of Adj Close')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Let's plot the training data to see if things changed a lot
train_raw_for_eda = train_raw.copy()
train_raw_for_eda['Date'] = pd.to_datetime(train_raw_for_eda['Date'])
train_raw_for_eda.sort_values('Date', inplace=True)

with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(15, 6))
    plt.plot(train_raw_for_eda['Date'], train_raw_for_eda['Adj Close'], linewidth=1, color='blue')
    plt.title('Training Data Time Series (Adjusted Close Price)')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.grid(True)
    # Optional: If you want to clearly mark notable periods or changes
    # plt.axvline(pd.Timestamp('2022-01-01'), color='red', linestyle='--', label='Notable Change')
    # plt.legend()
    plt.tight_layout()
    plt.show()
    
# Let's see the extreme events in the training set per weekday
train_raw_for_eda["Date"] = pd.to_datetime(train_raw_for_eda["Date"])
train_raw_for_eda["Day_of_Week"] = train_raw_for_eda["Date"].dt.day_name()
weekday_extreme_events = train_raw_for_eda.groupby('Day_of_Week')['Extreme_Event'].mean()
weekday_extreme_events = weekday_extreme_events.reindex(weekday_order)
weekday_extreme_events

# Same for the validation set
val_raw_for_eda["Date"] = pd.to_datetime(val_raw_for_eda["Date"])
val_raw_for_eda["Day_of_Week"] = val_raw_for_eda["Date"].dt.day_name()
weekday_extreme_events = val_raw_for_eda.groupby('Day_of_Week')['Extreme_Event_Pred'].mean()
weekday_extreme_events = weekday_extreme_events.reindex(weekday_order)
weekday_extreme_events

## Now let's also do something about the low precision
correct_preds = val_raw_for_eda[(val_raw_for_eda['Extreme_Event'] == 1) & (val_raw_for_eda['Correct_Pred_Flag'] == 1)]
incorrect_preds = val_raw_for_eda[(val_raw_for_eda['Extreme_Event'] == 0) & (val_raw_for_eda['Correct_Pred_Flag'] == 0)]

with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(14, 7))

    # Plot Adjusted Close price
    plt.plot(val_raw_for_eda['Date'], val_raw_for_eda['Adj Close'], linewidth=1, label='Adj Close')

    # Plot correct and incorrect predictions with different markers/colors
    plt.scatter(correct_preds['Date'], correct_preds['Adj Close'], color='green', label='Correct Predictions')
    plt.scatter(incorrect_preds['Date'], incorrect_preds['Adj Close'], color='red', label='Incorrect Predictions')

    # Enhancing plot readability
    plt.title('Adjusted Close Price with Extreme Events', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


"""
