# Libraries
import yfinance as yf
import pandas as pd
import pathlib
import logging

# Paths:
THIS_FILE = pathlib.Path(__file__)
PACKAGE_ROOT = THIS_FILE.parent.parent
INPUT_DATA = PACKAGE_ROOT / "data"
APPLE_STOCK_FILE_PATH = (
    INPUT_DATA / "apple_daily_stock_data.csv"
)  # Define the full file path

# Static Parameters
start_date = "2015-01-01"
end_date = "2024-02-01"
stock = "AAPL"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Helper Functions
def download_stock_data(
    ticker: str, start: str, end: str, save_path: str = APPLE_STOCK_FILE_PATH
) -> pd.DataFrame:
    """Download Apple stock data and save it as a CSV file."""
    logging.info(f"Downloading stock data for {ticker} from {start} to {end}")
    df = yf.download(ticker, start, end, auto_adjust=False)

    # To pandas DataFrame
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)  # Working with the index (Date) as a column

    # Flatten the column names, keeping only the first level
    df.columns = df.columns.get_level_values(0)

    # Reorder the columns
    column_order = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
    df = df[column_order]  # Reorder the columns

    # Save data
    df.to_csv(save_path, index=False)  # Now writing to a file, not a directory
    logging.info(f"Stock data saved to {save_path}")

    return df


### Step 1.1: Load the Data

dfr = download_stock_data(stock, start_date, end_date)

### Step 1.2: Calculate Daily Returns (and forward fill missing values)

# Calculating the Daily_Return
dfr["Daily_Return"] = (
    (dfr["Adj Close"] - dfr["Adj Close"].shift(1)) / dfr["Adj Close"].shift(1) * 100
)

### Step 1.3: Extreme_Event if Absolute Daily_Return > 2%

dfr["Extreme_Event"] = (dfr["Daily_Return"].abs() > 2).astype(int)

# Shift the Extreme_Event column by 1 day to avoid look-ahead
dfr["Extreme_Event"] = dfr["Extreme_Event"].shift(-1)

# Dropping NAs
dfr = dfr.dropna(inplace=False)

### Step 1.4: Split Data into Features and Target

# We are doing out of time validation, so we will split the data into training and testing sets based on the date
# Total number of days in dataset
min_date = dfr["Date"].min()
max_date = dfr["Date"].max()
days_in_dataset = (max_date - min_date).days

# Identify split points (sequential split to preserve time order)
split_date_train = min_date + pd.DateOffset(days=round(days_in_dataset * 0.7))
split_date_val = split_date_train + pd.DateOffset(days=round(days_in_dataset * 0.15))

# Split dataset
training = dfr[dfr["Date"] <= split_date_train]
validation = dfr[(dfr["Date"] > split_date_train) & (dfr["Date"] <= split_date_val)]
testing = dfr[dfr["Date"] > split_date_val]

training["Date"].max()
validation["Date"].max()
testing["Date"].max()

# Save the data
training.to_csv(INPUT_DATA / "training.csv", index=False)
validation.to_csv(INPUT_DATA / "validation.csv", index=False)
testing.to_csv(INPUT_DATA / "testing.csv", index=False)

logging.info(f"Data processing completed successfully, data saved under {INPUT_DATA}")
logging.info(f"Number of rows in the training set: {len(training)}")
logging.info(f"Number of rows in the validation set: {len(validation)}")
logging.info(f"Number of rows in the testing set: {len(testing)}")
