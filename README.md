# Churn Prediction Pipeline

## Overview

This project implements a machine learning pipeline to predict customer churn using a dataset of bank customer information. The pipeline includes data import, exploratory data analysis (EDA), feature engineering, model training, and evaluation. It uses Logistic Regression and Random Forest models to predict churn and generates visualizations for analysis, including feature importance and SHAP summary plots. A comprehensive test suite ensures the reliability of the pipeline.

The main script, `churn_library.py`, contains the core functionality, while `churn_script_logging_and_tests.py` provides unit tests to validate each function. Logging captures detailed information about pipeline execution and test results in a log file for traceability.

## Prerequisites

- **Python Version**: Python 3.6 or higher
- **Dependencies**: Install required packages using the provided `requirements.txt`:
  ```plaintext
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  joblib
  shap
  python-dotenv
  ```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Ensure the dataset file (`bank_data.csv`) is located in the `./data` directory.
   - Alternatively, set the `BANK_DATA_PATH` environment variable to point to your dataset:
     ```bash
     export BANK_DATA_PATH=/path/to/your/bank_data.csv
     ```

4. **Directory Structure**:
   The pipeline creates the following directories if they do not exist:
   - `./image`: Stores EDA and model evaluation plots.
   - `./models`: Stores trained model files.
   - `./logs`: Stores log files for pipeline execution and tests.

## Usage

### Running the Pipeline

To execute the full churn prediction pipeline (data import, EDA, feature engineering, and model training), run:
```bash
python churn_library.py
```

This will:
- Load the dataset from `BANK_DATA_PATH` (default: `./data/bank_data.csv`).
- Perform EDA and save plots to `./image` (e.g., `churn_histogram.png`, `correlation_heatmap.png`).
- Encode categorical features and create churn proportion columns.
- Perform feature engineering to prepare training and testing datasets.
- Train Logistic Regression and Random Forest models.
- Save trained models to `./models` (e.g., `rfc_model.pkl`, `logistic_model.pkl`, `scaler.pkl`).
- Generate evaluation plots, including classification reports, ROC curves, feature importance, and SHAP summary plots, saved to `./image`.

To skip feature engineering and model training (e.g., for EDA only), set the `RUN_FEATURE_ENGINEERING` environment variable to `false`:
```bash
export RUN_FEATURE_ENGINEERING=false
python churn_library.py
```

Logs for the pipeline execution are saved to `./logs/churn_script.log`.

### Running Tests

To validate the pipeline functions, run the test script:
```bash
python churn_script_logging_and_tests.py
```
or
```bash
ipython churn_script_logging_and_tests.py
```

This will execute unit tests for all functions in `churn_library.py`:
- `import_data`
- `perform_eda`
- `encoder_helper`
- `perform_feature_engineering`
- `train_models`
- `classification_report_image`
- `feature_importance_plot`

The test script validates:
- Successful execution with valid inputs.
- Error handling for invalid inputs (e.g., missing files, invalid columns).
- Correct output formats (e.g., DataFrame structures, file generation).
- Presence of expected output files (e.g., model files, plots).

Test results and detailed logs are saved to `./logs/churn_script.log`. To view the logs:
```bash
cat ./logs/churn_script.log
```

To run individual test cases, specify the test method:
```bash
python -m unittest churn_script_logging_and_tests.TestEDAPipeline.test_import
```

### Log File

The log file (`./logs/churn_script.log`) contains:
- **Info Messages**: Indicate successful execution of pipeline steps or tests, including DataFrame properties, created files, and model parameters (e.g., "Testing import_data: SUCCESS").
- **Error Messages**: Detail any failures with specific error messages and stack traces for traceability (e.g., "Testing perform_eda: Failed to execute - Churn column already exists in DataFrame").

## Pipeline Details

### Functions in `churn_library.py`

1. **import_data(pth)**: Loads a CSV file into a pandas DataFrame.
2. **perform_eda(df)**: Performs EDA, creates a Churn column, and generates visualizations (e.g., histograms, correlation heatmap).
3. **encoder_helper(df, category_lst, response)**: Encodes categorical columns by calculating churn proportions.
4. **perform_feature_engineering(df, response)**: Prepares features for modeling, including scaling and train-test split.
5. **train_models(X_train, X_test, y_train, y_test, scaler)**: Trains Logistic Regression and Random Forest models, saves models, and generates evaluation plots.
6. **classification_report_image(y_train, y_test, ...)**: Generates classification reports and ROC curves for model evaluation.
7. **feature_importance_plot(model, X_data, output_pth)**: Creates feature importance and SHAP summary plots for the Random Forest model.

### Output Files

- **EDA Plots** (in `./image`):
  - `churn_histogram.png`
  - `customer_age_histogram.png`
  - `marital_status_bar.png`
  - `total_trans_ct_histogram.png`
  - `correlation_heatmap.png`
- **Model Files** (in `./models`):
  - `rfc_model.pkl` (Random Forest)
  - `logistic_model.pkl` (Logistic Regression)
  - `scaler.pkl` (StandardScaler)
- **Evaluation Plots** (in `./image`):
  - `logistic_classification_report.png`
  - `random_forest_classification_report.png`
  - `roc_curve.png`
  - `feature_importance.png`
  - `shap_summary_plot.png`
- **Log File** (in `./logs`):
  - `churn_script.log`

## Testing Details

The test script (`churn_script_logging_and_tests.py`) includes unit tests for all functions in `churn_library.py`. Each test method:
- Verifies successful execution with valid inputs.
- Checks error handling for invalid inputs.
- Validates output formats and file generation.
- Logs detailed success and error messages to `./logs/churn_script.log`.

## Troubleshooting

- **FileNotFoundError**: Ensure `bank_data.csv` exists at the specified `BANK_DATA_PATH`.
- **Missing Dependencies**: Verify all packages in `requirements.txt` are installed.
- **Log File Issues**: Check write permissions for the `./logs` directory.
- **Test Failures**: Review `./logs/churn_script.log` for detailed error messages.

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the repository for bug reports or feature requests.

## License

This project is licensed under the MIT License.