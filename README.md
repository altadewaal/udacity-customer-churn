# Churn Prediction Pipeline

## Overview
This project implements a machine learning pipeline to predict customer churn using bank customer data. It processes a dataset to create a 'Churn' column, performs exploratory data analysis (EDA) to generate visualizations, encodes categorical features, trains Logistic Regression and Random Forest models, and produces evaluation metrics and plots, including feature importance and SHAP summary plots. Outputs are saved as images (`./image/`) and model files (`./models/`).

The pipeline is implemented in `churn_library.py`, and unit tests are provided in `churn_script_logging_and_tests.py`. Test execution logs are saved to `./logs/churn_test_script.log`.

## Project Structure
- `churn_library.py`: Main script containing the pipeline functions and execution logic.
- `churn_script_logging_and_tests.py`: Unit tests for validating pipeline functions.
- `data/`: Directory containing the input dataset (`bank_data.csv`).
- `image/`: Directory for EDA and model evaluation plots (e.g., `churn_histogram.png`, `roc_curve.png`).
- `models/`: Directory for saved models and scaler (e.g., `rfc_model.pkl`, `logistic_model.pkl`, `scaler.pkl`).
- `logs/`: Directory for test logs (`churn_test_script.log`).
- `.env`: Optional environment file for configuring data path and pipeline options.

## Dependencies
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `shap`, `joblib`, `python-dotenv`
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

Sample `requirements.txt`:
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
shap>=0.41.0
joblib>=1.2.0
python-dotenv>=1.0.0
```

## Setup
Clone the Repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

Install Dependencies:
```bash
pip install -r requirements.txt
```

Prepare the Dataset:
- Place the dataset (`bank_data.csv`) in the `./data/` directory.
- Alternatively, configure the dataset path in a `.env` file:
  ```
  BANK_DATA_PATH=./data/bank_data.csv
  RUN_FEATURE_ENGINEERING=true
  ```

Create Output Directories:
```bash
mkdir -p ./logs
chmod u+w ./logs
```

## Running the Pipeline
To execute the churn prediction pipeline:
```bash
python churn_library.py
```

**Inputs:**
- Dataset: Specified via `BANK_DATA_PATH` in `.env` (default: `./data/bank_data.csv`).
- Feature Engineering: Controlled via `RUN_FEATURE_ENGINEERING` in `.env` (default: true).

**Outputs:**
- EDA plots: `./image/churn_histogram.png`, `./image/customer_age_histogram.png`, `./image/marital_status_bar.png`, `./image/total_trans_ct_histogram.png`, `./image/correlation_heatmap.png`
- Model evaluation plots: `./image/logistic_classification_report.png`, `./image/random_forest_classification_report.png`, `./image/roc_curve.png`, `./image/feature_importance.png`, `./image/shap_summary_plot.png`
- Models: `./models/rfc_model.pkl`, `./models/logistic_model.pkl`, `./models/scaler.pkl`

If `RUN_FEATURE_ENGINEERING=false` in `.env`, the pipeline skips feature engineering and model training, producing only EDA outputs.

## Running Tests
To run unit tests for the pipeline:
```bash
python churn_script_logging_and_tests.py
```

Logs: Test execution details are saved to `./logs/churn_test_script.log`. View them with:
```bash
cat ./logs/churn_test_script.log
```

**Tests Cover:**
- Data import (`import_data`)
- EDA (`perform_eda`)
- Categorical encoding (`encoder_helper`)
- Feature engineering (`perform_feature_engineering`)
- Model training (`train_models`)
- Classification report image generation (`classification_report_image`)
- Feature importance plotting (`feature_importance_plot`)

## Environment Variables
Configure the pipeline using a `.env` file in the project root:
```
BANK_DATA_PATH=./data/bank_data.csv
RUN_FEATURE_ENGINEERING=true
```

- `BANK_DATA_PATH`: Path to the input CSV file (default: `./data/bank_data.csv`).
- `RUN_FEATURE_ENGINEERING`: Set to true to run feature engineering and model training, or false to run only EDA (default: true).

## Notes
- Ensure the dataset (`bank_data.csv`) has the expected columns (e.g., `Attrition_Flag`, `Customer_Age`, `Gender`, etc.) to avoid errors.
- The pipeline assumes a binary churn classification task, with `Attrition_Flag` mapped to `Churn` (0 for "Existing Customer", 1 for "Attrited Customer").
- Test logs (`./logs/churn_test_script.log`) provide detailed debugging information for test failures.
- The pipeline itself (`churn_library.py`) does not generate execution logs.

## Troubleshooting
- **Missing Dataset:** Verify `bank_data.csv` exists at the specified `BANK_DATA_PATH`:
  ```bash
  ls ./data/bank_data.csv
  ```
- **Missing Columns:** Check the test log for missing column errors:
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('./data/bank_data.csv'); print(df.columns)"
  ```
- **Image Generation Errors:** If `test_classification_report_image` fails, check logs for errors.
- **Permission Issues:** Ensure write access to `./logs/`, `./image/`, and `./models/`:
  ```bash
  chmod -R u+w ./logs ./image ./models
  ```
- **Test Failures:** Check `./logs/churn_test_script.log` for detailed error messages.
- **Dependency Errors:** Reinstall dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Contributing
- Fork the repository.
- Create a feature branch (`git checkout -b feature/your-feature`).
- Commit changes (`git commit -m "Add your feature"`).
- Push to the branch (`git push origin feature/your-feature`).
- Open a pull request.

## License
This project is licensed under the MIT License.

_Last updated: 11:53 AM SAST, Sunday, September 28, 2025_


