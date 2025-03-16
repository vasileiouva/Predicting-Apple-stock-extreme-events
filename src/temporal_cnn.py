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


### Step 3.3: Model Training
def main(
    evaluate: bool = False,
    resample: bool = False,
    save_results: bool = False,
    weight_minority: float = 1.0,
):
    """
    Main function to train and evaluate a Temporal CNN model.
    """
    # Load the csv data
    logging.info("Loading data")
    train_raw = pd.read_csv(TRAINING_DATA)
    val_raw = pd.read_csv(VALIDATION_DATA)

    ### Step 3.1: Input Preparation
    
    # Scaling the data
    # Neural Nets are sensitive to the scale of the input data. For random forest, we didn't need to rescale the data.
    logging.info("Scaling the data")
    train_scaled, val_scaled, scaler = scale_train_val(train_raw, val_raw, FEATURES)

    # Apply resampling if required
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
    """
    We use shuffling for training data as each input sequence already contains all necessary historical data.
    If we don't do that, the model may learn biases from sequential patterns in the dataset order.
    If we had an RNN (LSTM) or Transformer model, we wouldn't shuffle the training data.
    """
    df_train = prepare_data(train_scaled, BATCH_SIZE, shuffle=True)
    df_val = prepare_data(val_scaled, BATCH_SIZE, shuffle=False)
    num_features = len(df_train.dataset[0][0])
    sequence_length = N_DAYS

    ### Step 3.2: Model Architecture

    # Define Temporal CNN Model (Helper class)
    class TemporalCNN(nn.Module):
        def __init__(self, num_features, sequence_length, dropout_rate=DROPOUT_RATE):
            super(TemporalCNN, self).__init__()

            self.conv1 = nn.Conv1d(
                in_channels=num_features, out_channels=32, kernel_size=3, padding=1
            )
            self.conv2 = nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)

            self.flatten = nn.Flatten()
            self.fc = nn.Linear(64 * sequence_length, 2)
            # self.softmax = nn.Softmax(dim=1) - We omit softmax since CrossEntropyLoss expects raw logits.

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.flatten(x)
            x = self.fc(x)
            return x  # self.softmax(x)

    ### Step 3.3: Model Training
    torch.manual_seed(SEED)
    model = TemporalCNN(num_features, sequence_length).to(DEVICE)

    logging.info("Starting TCNN training")

    # Loss & Optimiser
    class_weights = torch.tensor([1.0, weight_minority], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    logging.info("Starting model training")
    best_val_loss = float("inf")
    patience_counter = 0
    min_delta = 0.001
    best_state_dict = None

    for epoch in range(EPOCHS):

        # Training Step
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

        # Validation Step
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

    best_metrics = report_metrics(labels, preds, "TCNN Model (Validation Set)")

    ### Step 3.4: Model Evaluation
    if evaluate:
        logging.info("Evaluating the model on the test set")
        test_raw = pd.read_csv(TESTING_DATA)

        # Scale test using the SAME scaler (fit on train)
        logging.info("Scaling the test data")
        test_scaled = test_raw.copy()
        test_scaled[FEATURES] = scaler.transform(test_raw[FEATURES])

        # Prepare DataLoader
        df_test = prepare_data(test_scaled, BATCH_SIZE, shuffle=False)

        # Inference on test
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in df_test:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        final_test_metrics = report_metrics(all_labels, all_preds, "TCNN (Test Set)")
        logging.info("Final test evaluation completed.")

        # Save results as CSV if requested
        if save_results:
            results_df = pd.DataFrame([best_metrics, final_test_metrics])
            filename = (
                "tcnn_model_comparison_resampled"
                if resample
                else "tcnn_model_comparison"
            )
            filename = (
                filename + f"_weighted_{weight_minority}.csv"
                if weight_minority != 1.0
                else filename + ".csv"
            )
            out_path = PACKAGE_ROOT / "data" / filename
            results_df.to_csv(out_path, index=False)
            logging.info(f"Model evaluation results saved to '{out_path.name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Temporal CNN.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model.")
    parser.add_argument(
        "--resample", action="store_true", help="Apply SMOTE resampling."
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results as CSV."
    )
    parser.add_argument(
        "--weight_minority",
        type=float,
        default=1.0,
        help="Weight to use for the positive class in CrossEntropyLoss.",
    )
    args = parser.parse_args()

    main(
        evaluate=args.evaluate,
        resample=args.resample,
        save_results=args.save_results,
        weight_minority=args.weight_minority,
    )
