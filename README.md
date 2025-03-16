# README.md

## **Apple Stock Price Extreme Events Prediction**

This project focuses on predicting **extreme events** in Apple stock prices using two machine learning models: a **Random Forest classifier** and a **Temporal Convolutional Neural Network (TCNN)**. An extreme event is defined as a **daily return exceeding ±2%** compared to the previous day's adjusted closing price.

---

## **Project Structure**

```
submission/
│
├── src/                 # Directory containing all Python executable scripts
│   ├── data_processing.py
│   ├── random_forest.py
│   ├── temporal_cnn.py
│   ├── model_evaluation.py
│   └── improvement.py
│
├── README.md            # Detailed instructions on how to run the code
│
├── pyproject.toml       # Poetry configuration file for dependency management
├── poetry.lock          # Poetry lock file for reproducibility
│
├── report.pdf           # A detailed report including model performance and analysis
│
└── data/                # Directory for datasets and generated results
    ├── training.csv
    ├── validation.csv
    └── testing.csv
```

---

## **Environment Setup**

This project uses **Poetry** for dependency management.

1. **Install Poetry** (if not already installed):
```bash
pip install poetry
```

2. **Set Up the Virtual Environment**:
```bash
poetry install
```

3. **Activate the Environment**:
```bash
poetry shell
```

---

## **How to Run Each Script**

### 1. **Data Preprocessing**
```bash
poetry run python src/data_processing.py
```
This script will load the raw data, preprocess it (including calculating daily returns and defining extreme events), and split it into training, validation, and testing datasets.

### 2. **Random Forest Model**
```bash
poetry run python src/random_forest.py --evaluate --save_results
```
- `--evaluate`: Evaluates the model on the test set.
- `--save_results`: Saves the performance results to the `/data/` folder.

### 3. **Temporal CNN Model**
```bash
poetry run python src/temporal_cnn.py --evaluate --save_results --resample --weight_minority
```
- `--evaluate`: Evaluates the TCNN model on the test set.
- `--save_results`: Saves the performance results to the `/data/` folder.
- `--resample`: Resamples the training data using SMOTE
- `--weight_minority`: Gives this weight to the minority class. Requires a numerical value for example
```bash
--weight_minority 5
```

### 4. **Model Evaluation**
```bash
poetry run python src/model_evaluation.py
```
- Compares the results from both models and outputs evaluation metrics.

### 5. **Improvement of TCNN**
```bash
poetry run python src/improvement.py --evaluate --save_results --hyperparameter_tuning --apply_feature_engineering --resample
```
- `--evaluate`: Evaluates the improved model.
- `--save_results`: Saves the results.
- `--hyperparameter_tuning`: Runs hyperparameter tuning for the TCNN model.
- `--apply_feature_engineering`: Applies additional feature engineering techniques.
- `--resample`: Resamples the training data using SMOTE
- `--save_results`: Saves the performance results to the `/data/` folder.

This script also contains an appendix with some EDA code for better understanding of data patterns and model insights.

---

## **Key Features**

- **Data Preprocessing**: Calculates daily returns, defines extreme events, and prepares data for model training.
- **Random Forest Model**: Trains and evaluates a Random Forest classifier using 10-day historical sequences.
- **Temporal CNN Model**: Builds, trains, and evaluates a TCNN to predict extreme events using historical sequences.
- **Model Evaluation**: Compares model performances using confusion matrices and classification metrics.
- **Improvements**:
  - **Feature Engineering**: Adds advanced features like SMA, EMA, Bollinger Bands, ATR, VWAP, and lag features.
  - **Hyperparameter Tuning**: Explores various configurations to optimize model performance.

---

## **Expected Outputs**

- Model performance metrics (Accuracy, Precision, Recall, F1-Score).
- Confusion matrices for both models.
- CSV files containing results and best hyperparameters.
- Detailed analysis and comparison included in `report.pdf`.

---

## **Dependencies**

All dependencies are listed in `pyproject.toml` and `poetry.lock` to ensure reproducibility. Key libraries include:
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `imblearn`
- `matplotlib`
- `Python 3.10.12`
- `yfinance`

---

## **Reproducibility**

To reproduce the results:
1. Install dependencies using Poetry.
2. Run the preprocessing and modeling scripts as outlined.

---

## **Report**

  - The `report.pdf` includes:
  - A detailed analysis of each model's performance.
  - Comparison between the Random Forest and TCNN models.
  - Discussion on model improvement strategies.
  - Reflection on the challenges of predicting extreme stock price movements.

---

## **Future Improvements**

  - Add Unit Tests
  - Make the code more modular
  - Add more features
  - Deeper Error Analysis

---

##  **Contact**
For any questions or issues, please contact Vas on ```vasileiou.va@gmail.com```.

---
