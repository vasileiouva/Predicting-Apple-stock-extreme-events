# Libraries
import subprocess
import logging
import pathlib

# Paths
THIS_FILE = pathlib.Path(__file__)
PACKAGE_ROOT = THIS_FILE.parent.parent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Helper Functions
def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=True, cwd=PACKAGE_ROOT)
    print(f"Finished: {command}")
    return result

if __name__ == "__main__":
    logging.info("PREPROCESSING DATA")
    run_command("poetry run python src/data_processing.py")
    logging.info("RANDOM FOREST MODEL WITHOUT RESAMPLING")
    run_command("poetry run python src/random_forest.py --evaluate --save_results")
    logging.info("RANDOM FOREST MODEL WIH RESAMPLING")
    run_command("poetry run python src/random_forest.py --evaluate --resample --save_results")
    logging.info("TCNN MODEL WITHOUT RESAMPLING")
    run_command("poetry run python src/temporal_cnn.py --evaluate --save_results")
    logging.info("TCNN MODEL WIH RESAMPLING")
    run_command("poetry run python src/temporal_cnn.py --evaluate --resample --save_results")
    logging.info("TCNN MODEL WITH A WEIGHTED LOSS FUNCTION")
    run_command("poetry run python src/temporal_cnn.py --evaluate --weight_minority 4.06 --save_results")
