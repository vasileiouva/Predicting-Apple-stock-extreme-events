# Libraries
import argparse
import pandas as pd
import pathlib
import logging
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Paths:
THIS_FILE = pathlib.Path(__file__)
PACKAGE_ROOT = THIS_FILE.parent.parent
TRAINING_DATA = PACKAGE_ROOT / "data" / "training.csv"
VALIDATION_DATA = PACKAGE_ROOT / "data" / "validation.csv"
TESTING_DATA = PACKAGE_ROOT / "data" / "testing.csv"

# Static Parameters
N_DAYS = 10

# Setting a seeds for full reproducibility
SEED = 2187
np.random.seed(SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Helper Functions
def create_lag_features(
    df: pd.DataFrame,
    n_days: int = N_DAYS,
    feature_columns: list = ["Open", "High", "Low", "Close", "Volume", "Daily_Return"],
) -> pd.DataFrame:
    """
    This function take a Pandas dataframe and creates N_DAYS lagged variables as features for each element in feature_columns.
    """
    logging.info(f"Creating lagged features for {n_days} days")
    X = []
    y = []

    for i in range(len(df) - N_DAYS):
        # Extract past N_DAYS of features and flatten
        past_features = df.iloc[i : i + N_DAYS][feature_columns].values.flatten()

        # Target: "Extreme_Event" on day i + n_days
        target = df.iloc[i + N_DAYS]["Extreme_Event"]

        X.append(past_features)
        y.append(target)

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Convert to DataFrame for convenience
    columns = [f"{col}_t-{i}" for i in range(N_DAYS, 0, -1) for col in feature_columns]
    X_df = pd.DataFrame(X, columns=columns)
    y_df = pd.Series(y, name="Extreme_Event")

    # Final dataset
    df_lagged = pd.concat([X_df, y_df], axis=1)

    logging.info("Finished Creating lagged features")

    return df_lagged


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


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring_metric: str = "f1",
    cv_folds: int = 5,
    resample: bool = False,
) -> RandomForestClassifier:
    """
    Perform hyperparameter tuning for a Random Forest classifier
    """
    pipeline_steps = []
    if resample:
        pipeline_steps.append(("smote", SMOTE(random_state=SEED)))
    pipeline_steps.append(("rf", RandomForestClassifier(random_state=SEED)))

    pipeline = Pipeline(steps=pipeline_steps)

    param_grid = {
        "rf__n_estimators": [50, 100, 200, 300],
        "rf__max_depth": [5, 10, 20, None],
        "rf__max_features": ["sqrt", "log2"],
    }

    # Initialise GridSearchCV
    # I will use stratified cross-validation to preserve the class distribution across folds
    cv_logic = TimeSeriesSplit(n_splits=cv_folds)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_logic,
        scoring=scoring_metric,
        n_jobs=-1,
        verbose=0,
    )

    # Perform Grid Search
    logging.info("Starting hyperparameter tuning with GridSearchCV")
    grid_search.fit(X_train, y_train)

    # Retrieve best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logging.info(f"Best Hyperparameters: {best_params}")
    logging.info(
        f"Best Model F1-Score achieved while model training: {grid_search.best_score_ * 100:.2f}%"
    )

    return best_model


### Step 2.1: Model Training
def main(evaluate: bool = False, resample: bool = False, save_results: bool = False):
    """
    Main function to train and evaluate a Random Forest model.
    """
    logging.info("Starting Random Forest model training")

    # Read the training data
    train_raw = pd.read_csv(TRAINING_DATA)
    val_raw = pd.read_csv(VALIDATION_DATA)

    # Create the lagged data frame
    df_train_lagged = create_lag_features(train_raw)
    df_val_lagged = create_lag_features(val_raw)
    # df_lagged.to_csv(PACKAGE_ROOT / "data" / "check.csv", index=False)

    logging.info(
        f"The extreme events in training df are {df_train_lagged['Extreme_Event'].mean() * 100:.2f}% of the training data"
    )

    logging.info(
        f"The extreme events in validation df are {df_val_lagged['Extreme_Event'].mean() * 100:.2f}% of the validation data"
    )

    ### Step 2.1: Model Training

    # Split into features and target
    X_train, y_train = (
        df_train_lagged.drop(columns=["Extreme_Event"]),
        df_train_lagged["Extreme_Event"],
    )
    X_val, y_val = (
        df_val_lagged.drop(columns=["Extreme_Event"]),
        df_val_lagged["Extreme_Event"],
    )

    # Train Baseline Model (without resampling or HP tuning)
    rf_baseline = RandomForestClassifier(random_state=SEED)
    rf_baseline.fit(X_train, y_train)
    y_val_pred_baseline = rf_baseline.predict(X_val)
    baseline_metrics = report_metrics(
        y_val, y_val_pred_baseline, "Random Forest Baseline (Validation Set)"
    )

    # Tune Hyperparameters with optional resampling inside pipeline
    best_rf = tune_random_forest(X_train, y_train, resample=resample)
    y_val_pred_best = best_rf.predict(X_val)
    best_metrics = report_metrics(
        y_val, y_val_pred_best, "Random Forest Best Model (Validation Set)"
    )

    ### Step 2.2: Model Evaluation

    # Evaluate Model on the test set
    if evaluate:

        logging.info("Evaluating best Random Forest on test set")

        # Get the test data ready
        test_raw = pd.read_csv(TESTING_DATA)
        df_test_lagged = create_lag_features(test_raw)
        X_test, y_test = (
            df_test_lagged.drop(columns=["Extreme_Event"]),
            df_test_lagged["Extreme_Event"],
        )

        logging.info(
            "The extreme events in test df are {df_test_lagged['Extreme_Event'].mean() * 100:.2f}% of the test data"
        )

        logging.info("Merging train + val for final training with best hyperparameters")
        # Train final model on train + validation
        X_train_consolidated = pd.concat([X_train, X_val])
        y_train_consolidated = pd.concat([y_train, y_val])

        # Train Baseline Model on train + validation
        rf_baseline.fit(X_train_consolidated, y_train_consolidated)

        # Evaluate baseline model on test set
        y_test_pred_baseline = rf_baseline.predict(X_test)
        baseline_metrics_test = report_metrics(
            y_test, y_test_pred_baseline, "Random Forest Baseline Model (Test Set)"
        )

        # Retrieve best hyperparameters
        best_params = best_rf.get_params()

        # Filter out only parameters relevant to the RandomForestClassifier
        rf_params = {
            key.replace("rf__", ""): value
            for key, value in best_params.items()
            if key.startswith("rf__")
        }

        # Final pipeline
        if resample:
            final_pipeline = Pipeline(
                [
                    ("smote", SMOTE(random_state=SEED)),
                    ("rf", RandomForestClassifier(**rf_params)),
                ]
            )
        else:
            final_pipeline = RandomForestClassifier(**rf_params)

        final_pipeline.fit(X_train_consolidated, y_train_consolidated)
        y_test_pred = final_pipeline.predict(X_test)

        # Evaluate on test set
        logging.info("Evaluating best Random Forest on test set")
        y_test_pred = final_pipeline.predict(X_test)
        best_metrics_test = report_metrics(
            y_test, y_test_pred, "Random Forest Final Model (Test Set)"
        )

        # Save results as CSV
        if save_results:
            results_df = pd.DataFrame(
                [
                    baseline_metrics,
                    best_metrics,
                    baseline_metrics_test,
                    best_metrics_test,
                ]
            )
            filename = (
                "rf_model_comparison_resampled.csv"
                if resample
                else "rf_model_comparison.csv"
            )
            out_path = PACKAGE_ROOT / "data" / filename
            results_df.to_csv(out_path, index=False)
            logging.info(f"Model evaluation results saved to '{out_path.name}'.")

    logging.info("Task completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest model."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the model on test set"
    )
    parser.add_argument(
        "--resample", action="store_true", help="Apply SMOTE resampling."
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results as CSV."
    )
    args = parser.parse_args()

    main(evaluate=args.evaluate, resample=args.resample, save_results=args.save_results)
