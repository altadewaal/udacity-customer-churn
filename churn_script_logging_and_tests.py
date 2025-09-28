"""
Unit Tests for Churn Prediction Pipeline

This module provides a comprehensive test suite for the churn prediction pipeline implemented in churn_library.py. It includes unit tests for all functions to validate their functionality, error handling, and output correctness. The tests cover data import, exploratory data analysis (EDA), categorical encoding, feature engineering, model training, classification report generation, and feature importance plotting. Test results and detailed logs are saved to './logs/churn_test_script.log' for traceability.

Classes:
- TestEDAPipeline: Contains test methods for each function in churn_library.py, including input validation, output checks, and error handling.
"""
import unittest
import logging
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import churn_library as cls

# Load .env file
load_dotenv()

# Get data file path from environment variable, with fallback
DATA_FILE_PATH = os.environ.get('BANK_DATA_PATH', './data/bank_data.csv')

# Create logs directory and configure logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

class TestEDAPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures by loading the DataFrame."""
        try:
            self.df = cls.import_data(DATA_FILE_PATH)
            logging.info(f"setUp: Successfully loaded DataFrame from {DATA_FILE_PATH}")
            logging.info(f"setUp: DataFrame columns: {self.df.columns.tolist()}")
        except FileNotFoundError as err:
            logging.error(f"setUp: Failed to load DataFrame from {DATA_FILE_PATH} - File not found")
            raise err

    def test_import(self):
        """Test the import_data function."""
        try:
            df = cls.import_data(DATA_FILE_PATH)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error(f"Testing import_data: The file {DATA_FILE_PATH} wasn't found")
            raise err
        try:
            self.assertGreater(df.shape[0], 0, "DataFrame has no rows")
            self.assertGreater(df.shape[1], 0, "DataFrame has no columns")
            logging.info("Testing import_data: Shape check passed")
        except AssertionError as err:
            logging.error(f"Testing import_data: The file doesn't appear to have rows and columns - {str(err)}")
            raise err

    def test_eda(self):
        """Test the perform_eda function."""
        try:
            # Ensure Churn column doesn't exist
            df_eda = self.df.copy()
            if 'Churn' in df_eda.columns:
                df_eda = df_eda.drop('Churn', axis=1)
            # Log DataFrame properties
            logging.info(f"Testing perform_eda: DataFrame shape: {df_eda.shape}")
            missing_values = df_eda.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                logging.info(f"Testing perform_eda: Columns with missing values:\n{missing_values.to_string()}")
            else:
                logging.info("Testing perform_eda: No columns with missing values")
            logging.info(f"Testing perform_eda: DataFrame description:\n{df_eda.describe().to_string()}")

            # Run the perform_eda function
            cls.perform_eda(df_eda)
            logging.info("Testing perform_eda: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing perform_eda: Failed to execute - {str(err)}")
            raise err

        try:
            # Check if the Churn column was added
            self.assertIn('Churn', df_eda.columns, "Churn column not found in DataFrame")
            self.assertTrue(df_eda['Churn'].isin([0, 1]).all(), "Churn column contains invalid values")
            logging.info("Testing perform_eda: Churn column added successfully")
        except AssertionError as err:
            logging.error(f"Testing perform_eda: Churn column check failed - {str(err)}")
            raise err

        try:
            # Check if the image folder was created
            self.assertTrue(os.path.exists('image'), "Image folder was not created")
            logging.info("Testing perform_eda: Image folder created successfully")
        except AssertionError as err:
            logging.error(f"Testing perform_eda: Image folder creation failed - {str(err)}")
            raise err

        try:
            # List of expected plot files
            expected_plots = [
                'churn_histogram.png',
                'customer_age_histogram.png',
                'marital_status_bar.png',
                'total_trans_ct_histogram.png',
                'correlation_heatmap.png'
            ]
            # Check if all expected plot files exist in the image folder
            for plot in expected_plots:
                plot_path = os.path.join('image', plot)
                self.assertTrue(os.path.isfile(plot_path), f"Plot file {plot} not found")
            logging.info("Testing perform_eda: All expected plot files created successfully")
        except AssertionError as err:
            logging.error(f"Testing perform_eda: Plot file check failed - {str(err)}")
            raise err

        # Test case: Ensure perform_eda raises error if Churn column exists
        try:
            # Create a copy with Churn column
            df_with_churn = self.df.copy()
            df_with_churn['Churn'] = 1  # Dummy Churn column
            with self.assertRaises(ValueError) as context:
                cls.perform_eda(df_with_churn)
            self.assertEqual(str(context.exception), "Churn column already exists in DataFrame")
            logging.info("Testing perform_eda: Correctly raised ValueError for existing Churn column")
        except AssertionError as err:
            logging.error(f"Testing perform_eda: Failed to raise ValueError for existing Churn column - {str(err)}")
            raise err

    def test_encoder_helper(self):
        """Test the encoder_helper function."""
        # Define categorical columns
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        # Expected new columns
        new_columns = [f"{col}_Churn" for col in cat_columns]

        # Prepare DataFrame with perform_eda
        try:
            # Use a fresh copy to avoid modifying self.df
            df_eda = self.df.copy()
            if 'Churn' in df_eda.columns:
                df_eda = df_eda.drop('Churn', axis=1)
            cls.perform_eda(df_eda)
            logging.info("Testing encoder_helper: Successfully ran perform_eda")
            logging.info(f"Testing encoder_helper: df_eda columns after perform_eda: {df_eda.columns.tolist()}")
        except Exception as err:
            logging.error(f"Testing encoder_helper: Failed to run perform_eda - {str(err)}")
            raise err

        # Test case 1: Normal execution
        try:
            # Verify categorical columns exist
            missing_cat_cols = [col for col in cat_columns if col not in df_eda.columns]
            self.assertTrue(
                not missing_cat_cols,
                f"Categorical columns missing in df_eda: {missing_cat_cols}"
            )
            df_output = cls.encoder_helper(df_eda, cat_columns, response='Churn')
            logging.info("Testing encoder_helper: SUCCESS - Function executed without errors")
            logging.info(f"Testing encoder_helper: df_output columns: {df_output.columns.tolist()}")
        except Exception as err:
            logging.error(f"Testing encoder_helper: Failed to execute - {str(err)}")
            raise err

        try:
            # Check if output is a DataFrame
            self.assertIsInstance(df_output, pd.DataFrame, "Output is not a pandas DataFrame")
            # Check if all original columns are preserved
            self.assertTrue(
                set(df_eda.columns).issubset(set(df_output.columns)),
                "Output DataFrame does not contain all original columns"
            )
            # Check if new churn columns were added
            for col in new_columns:
                self.assertIn(col, df_output.columns, f"Encoded column {col} not found")
                self.assertTrue(
                    df_output[col].between(0, 1).all(),
                    f"Column {col} contains invalid proportions"
                )
                self.assertTrue(
                    df_output[col].notnull().all(),
                    f"Column {col} contains null values"
                )
            # Verify original columns are unchanged
            for col in df_eda.columns:
                try:
                    pd.testing.assert_series_equal(
                        df_output[col],
                        df_eda[col],
                        check_names=False,
                        check_dtype=False
                    )
                except AssertionError:
                    self.fail(f"Column {col} was modified unexpectedly")
            logging.info("Testing encoder_helper: Output DataFrame validated successfully")
        except AssertionError as err:
            logging.error(f"Testing encoder_helper: Validation failed - {str(err)}")
            raise err

        # Test case 2: Invalid response column
        try:
            with self.assertRaises(ValueError) as context:
                cls.encoder_helper(df_eda, cat_columns, response='Invalid_Column')
            error_msg = str(context.exception)
            self.assertTrue(
                error_msg.startswith("Response column 'Invalid_Column' not found in DataFrame"),
                f"Expected error message to start with 'Response column Invalid_Column not found', got: {error_msg}"
            )
            logging.info("Testing encoder_helper: Correctly raised ValueError for invalid response column")
        except AssertionError as err:
            logging.error(f"Testing encoder_helper: Failed to raise ValueError for invalid response - {str(err)}")
            raise err

        # Test case 3: Invalid categorical column
        try:
            invalid_cat_columns = ['Invalid_Column']
            with self.assertRaises(ValueError) as context:
                cls.encoder_helper(df_eda, invalid_cat_columns, response='Churn')
            error_msg = str(context.exception)
            self.assertTrue(
                error_msg.startswith("Categorical column 'Invalid_Column' not found in DataFrame"),
                f"Expected error message to start with 'Categorical column Invalid_Column not found', got: {error_msg}"
            )
            logging.info("Testing encoder_helper: Correctly raised ValueError for invalid categorical column")
        except AssertionError as err:
            logging.error(f"Testing encoder_helper: Failed to raise ValueError for invalid categorical column - {str(err)}")
            raise err

    def test_perform_feature_engineering(self):
        """Test the perform_feature_engineering function."""
        # Define expected keep_cols
        keep_cols = [
            'Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn'
        ]

        # Prepare DataFrame with perform_eda and encoder_helper
        try:
            df_eda = self.df.copy()
            if 'Churn' in df_eda.columns:
                df_eda = df_eda.drop('Churn', axis=1)
            cls.perform_eda(df_eda)
            logging.info(f"Testing perform_feature_engineering: df_eda columns after perform_eda: {df_eda.columns.tolist()}")
            cat_columns = [
                'Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
            ]
            df_encoded = cls.encoder_helper(df_eda, cat_columns, response='Churn')
            logging.info(f"Testing perform_feature_engineering: df_encoded columns: {df_encoded.columns.tolist()}")
        except Exception as err:
            logging.error(f"Testing perform_feature_engineering: Failed to prepare encoded DataFrame - {str(err)}")
            raise err

        # Test case 1: Normal execution
        try:
            X_train, X_test, y_train, y_test, scaler = cls.perform_feature_engineering(df_encoded, response='Churn')
            logging.info("Testing perform_feature_engineering: SUCCESS - Function executed without errors")
            logging.info(f"Testing perform_feature_engineering: X_train columns: {X_train.columns.tolist()}")
        except Exception as err:
            logging.error(f"Testing perform_feature_engineering: Failed to execute - {str(err)}")
            raise err

        try:
            # Check if X_train and X_test contain exactly the keep_cols
            self.assertEqual(
                sorted(X_train.columns.tolist()),
                sorted(keep_cols),
                "X_train does not contain exactly the expected keep_cols"
            )
            self.assertEqual(
                sorted(X_test.columns.tolist()),
                sorted(keep_cols),
                "X_test does not contain exactly the expected keep_cols"
            )
            # Check if y_train and y_test are pandas Series
            self.assertIsInstance(y_train, pd.Series, "y_train is not a pandas Series")
            self.assertIsInstance(y_test, pd.Series, "y_test is not a pandas Series")
            # Check if y_train and y_test match the response column
            self.assertTrue(
                pd.concat([y_train, y_test]).sort_index().equals(df_encoded['Churn'].sort_index()),
                "y_train and y_test do not match the response column"
            )
            # Check train-test split sizes
            self.assertGreater(X_train.shape[0], 0, "X_train has no rows")
            self.assertGreater(X_test.shape[0], 0, "X_test has no rows")
            self.assertGreater(len(y_train), 0, "y_train has no values")
            self.assertGreater(len(y_test), 0, "y_test has no values")
            total_rows = X_train.shape[0] + X_test.shape[0]
            self.assertEqual(
                total_rows,
                df_encoded.shape[0],
                "Train/test split does not match original DataFrame size"
            )
            # Check approximate 70-30 split
            expected_train_ratio = 0.7
            actual_train_ratio = X_train.shape[0] / df_encoded.shape[0]
            self.assertAlmostEqual(
                actual_train_ratio,
                expected_train_ratio,
                places=1,
                msg=f"Expected train ratio ~{expected_train_ratio}, got {actual_train_ratio}"
            )
            # Check no missing values in X_train and X_test
            self.assertFalse(X_train.isnull().any().any(), "X_train contains missing values")
            self.assertFalse(X_test.isnull().any().any(), "X_test contains missing values")
            # Check if scaler is a StandardScaler object
            self.assertIsInstance(scaler, StandardScaler, "Scaler is not a StandardScaler object")
            logging.info("Testing perform_feature_engineering: Output validated successfully")
        except AssertionError as err:
            logging.error(f"Testing perform_feature_engineering: Validation failed - {str(err)}")
            raise err

        # Test case 2: Invalid response column
        try:
            with self.assertRaises(KeyError) as context:
                cls.perform_feature_engineering(df_encoded, response='Invalid_Column')
            self.assertTrue(
                'Invalid_Column' in str(context.exception),
                f"Expected KeyError for missing response column, got: {str(context.exception)}"
            )
            logging.info("Testing perform_feature_engineering: Correctly raised KeyError for invalid response column")
        except AssertionError as err:
            logging.error(f"Testing perform_feature_engineering: Failed to raise KeyError for invalid response - {str(err)}")
            raise err

        # Test case 3: Missing keep_cols
        try:
            # Create a DataFrame missing one numerical column
            df_missing = df_encoded.drop(columns=['Customer_Age'])
            with self.assertRaises(KeyError) as context:
                cls.perform_feature_engineering(df_missing, response='Churn')
            self.assertTrue(
                'Customer_Age' in str(context.exception),
                f"Expected KeyError for missing Customer_Age, got: {str(context.exception)}"
            )
            logging.info("Testing perform_feature_engineering: Correctly raised KeyError for missing keep_cols")
        except AssertionError as err:
            logging.error(f"Testing perform_feature_engineering: Failed to raise KeyError for missing keep_cols - {str(err)}")
            raise err

    def test_train_models(self):
        """Test the train_models function."""
        try:
            # Prepare DataFrame with perform_eda and encoder_helper
            df_eda = self.df.copy()
            if 'Churn' in df_eda.columns:
                df_eda = df_eda.drop('Churn', axis=1)
            cls.perform_eda(df_eda)
            logging.info(f"Testing train_models: df_eda columns after perform_eda: {df_eda.columns.tolist()}")
            cat_columns = [
                'Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
            ]
            df_encoded = cls.encoder_helper(df_eda, cat_columns, response='Churn')
            logging.info(f"Testing train_models: df_encoded columns: {df_encoded.columns.tolist()}")
            # Perform feature engineering
            X_train, X_test, y_train, y_test, scaler = cls.perform_feature_engineering(df_encoded, response='Churn')
            logging.info(f"Testing train_models: X_train columns: {X_train.columns.tolist()}")
            cls.train_models(X_train, X_test, y_train, y_test, scaler)
            logging.info("Testing train_models: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing train_models: Failed to execute - {str(err)}")
            raise err

        try:
            expected_files = [
                './models/rfc_model.pkl',
                './models/logistic_model.pkl',
                './models/scaler.pkl',
                './image/logistic_classification_report.png',
                './image/random_forest_classification_report.png',
                './image/roc_curve.png',
                './image/feature_importance.png',
                './image/shap_summary_plot.png'
            ]
            for file_path in expected_files:
                self.assertTrue(os.path.isfile(file_path), f"File {file_path} not found")
            logging.info("Testing train_models: All expected files created successfully")
        except AssertionError as err:
            logging.error(f"Testing train_models: File check failed - {str(err)}")
            raise err

    def test_classification_report_image(self):
        """Test the classification_report_image function."""
        try:
            # Prepare data for testing
            df_eda = self.df.copy()
            if 'Churn' in df_eda.columns:
                df_eda = df_eda.drop('Churn', axis=1)
            cls.perform_eda(df_eda)
            cat_columns = [
                'Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
            ]
            df_encoded = cls.encoder_helper(df_eda, cat_columns, response='Churn')
            X_train, X_test, y_train, y_test, scaler = cls.perform_feature_engineering(df_encoded, response='Churn')
            
            # Train a simple model to generate predictions
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_train_preds = model.predict(X_train)
            y_test_preds = model.predict(X_test)
            y_test_probs = model.predict_proba(X_test)[:, 1]
            
            # Run the classification_report_image function
            cls.classification_report_image(
                y_train, y_test,
                y_train_preds, y_train_preds,  # Using same predictions for LR and RF for simplicity
                y_test_preds, y_test_preds,
                y_test_probs, y_test_probs
            )
            logging.info("Testing classification_report_image: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing classification_report_image: Failed to execute - {str(err)}")
            raise err

        try:
            expected_files = [
                './image/logistic_classification_report.png',
                './image/random_forest_classification_report.png',
                './image/roc_curve.png'
            ]
            for file_path in expected_files:
                self.assertTrue(os.path.isfile(file_path), f"File {file_path} not found")
            logging.info("Testing classification_report_image: All expected plot files created successfully")
        except AssertionError as err:
            logging.error(f"Testing classification_report_image: Plot file check failed - {str(err)}")
            raise err

        # Test case: Invalid input (mismatched lengths for probability arrays)
        try:
            y_test_probs_invalid = y_test_probs[:-1]  # Shorten y_test_probs by one
            with self.assertRaises(ValueError) as context:
                cls.classification_report_image(
                    y_train, y_test,
                    y_train_preds, y_train_preds,
                    y_test_preds, y_test_preds,
                    y_test_probs_invalid, y_test_probs
                )
            self.assertTrue(
                "Mismatched lengths between y_test and probability arrays" in str(context.exception),
                f"Expected ValueError for mismatched lengths, got: {str(context.exception)}"
            )
            logging.info("Testing classification_report_image: Correctly raised ValueError for invalid input")
        except AssertionError as err:
            logging.error(f"Testing classification_report_image: Failed to raise ValueError for invalid input - {str(err)}")
            raise err

    def test_feature_importance_plot(self):
        """Test the feature_importance_plot function."""
        try:
            # Prepare data for testing
            df_eda = self.df.copy()
            if 'Churn' in df_eda.columns:
                df_eda = df_eda.drop('Churn', axis=1)
            cls.perform_eda(df_eda)
            cat_columns = [
                'Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
            ]
            df_encoded = cls.encoder_helper(df_eda, cat_columns, response='Churn')
            X_train, X_test, y_train, y_test, scaler = cls.perform_feature_engineering(df_encoded, response='Churn')
            
            # Train a simple RandomForest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            # Run the feature_importance_plot function
            cls.feature_importance_plot(model, X_test, './image/feature_importance_test.png')
            logging.info("Testing feature_importance_plot: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing feature_importance_plot: Failed to execute - {str(err)}")
            raise err

        try:
            expected_files = [
                './image/feature_importance_test.png',
                './image/shap_summary_plot.png'
            ]
            for file_path in expected_files:
                self.assertTrue(os.path.isfile(file_path), f"File {file_path} not found")
            logging.info("Testing feature_importance_plot: All expected plot files created successfully")
        except AssertionError as err:
            logging.error(f"Testing feature_importance_plot: Plot file check failed - {str(err)}")
            raise err

        # Test case: Invalid model (no feature_importances_)
        try:
            from sklearn.linear_model import LinearRegression
            invalid_model = LinearRegression()
            invalid_model.fit(X_train, y_train)
            with self.assertRaises(AttributeError) as context:
                cls.feature_importance_plot(invalid_model, X_test, './image/feature_importance_invalid.png')
            self.assertTrue(
                'feature_importances_' in str(context.exception),
                f"Expected AttributeError for missing feature_importances_, got: {str(context.exception)}"
            )
            logging.info("Testing feature_importance_plot: Correctly raised AttributeError for invalid model")
        except AssertionError as err:
            logging.error(f"Testing feature_importance_plot: Failed to raise AttributeError for invalid model - {str(err)}")
            raise err

if __name__ == '__main__':
    unittest.main()