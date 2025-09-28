"""
Unit tests for churn_library.py with logging to './logs/churn_test_script.log'.

This module tests the functionality of the churn prediction pipeline, including
data import, EDA, categorical encoding, feature engineering, model training, and
visualization functions. Tests are logged to './logs/churn_test_script.log' for
debugging and verification.

Classes:
    TestChurnPrediction: Unit test class for churn_library functions.
"""
import os
import logging
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import churn_library
from unittest.mock import patch

# Configure logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(
    filename='./logs/churn_test_script.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

class TestChurnPrediction(unittest.TestCase):
    """
    Test case class for churn prediction pipeline functions.

    Attributes:
        data_frame (pd.DataFrame): Test DataFrame loaded from CSV.
        test_model (RandomForestClassifier): Mock model for testing.
    """
    def setUp(self) -> None:
        """
        Set up test fixtures before each test method.

        Loads the dataset and initializes a mock model.
        """
        try:
            self.data_frame = churn_library.import_data('./data/bank_data.csv')
            self.test_model = RandomForestClassifier(random_state=42)
            logging.info("setUp: Successfully loaded DataFrame and initialized model")
        except Exception as err:
            logging.error("setUp: Failed - %s", str(err))
            raise err

    def test_import_data(self) -> None:
        """
        Test the import_data function for correct DataFrame loading.

        Verifies that the DataFrame is non-empty and has expected columns.
        """
        try:
            self.assertIsInstance(self.data_frame, pd.DataFrame)
            self.assertGreater(len(self.data_frame), 0, "DataFrame is empty")
            self.assertIn('Attrition_Flag', self.data_frame.columns)
            logging.info("Testing import_data: SUCCESS")
        except AssertionError as err:
            logging.error("Testing import_data: FAILED - %s", str(err))
            raise err

    def test_perform_eda(self) -> None:
        """
        Test the perform_eda function for generating visualizations.

        Checks that the Churn column is created and plots are saved.
        """
        try:
            test_df = self.data_frame.copy()
            churn_library.perform_eda(test_df)
            self.assertIn('Churn', test_df.columns)
            self.assertTrue(os.path.exists('./image/churn_histogram.png'))
            self.assertTrue(os.path.exists('./image/customer_age_histogram.png'))
            self.assertTrue(os.path.exists('./image/marital_status_bar.png'))
            self.assertTrue(os.path.exists('./image/total_trans_ct_histogram.png'))
            self.assertTrue(os.path.exists('./image/correlation_heatmap.png'))
            logging.info("Testing perform_eda: SUCCESS")
        except AssertionError as err:
            logging.error("Testing perform_eda: FAILED - %s", str(err))
            raise err

    def test_encoder_helper(self) -> None:
        """
        Test the encoder_helper function for categorical encoding.

        Verifies that new encoded columns are created.
        """
        try:
            test_df = self.data_frame.copy()
            test_df['Churn'] = test_df['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
            cat_columns = [
                'Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
            ]
            encoded_df = churn_library.encoder_helper(test_df, cat_columns, 'Churn')
            for col in cat_columns:
                self.assertIn(f"{col}_Churn", encoded_df.columns)
            logging.info("Testing encoder_helper: SUCCESS")
        except AssertionError as err:
            logging.error("Testing encoder_helper: FAILED - %s", str(err))
            raise err
        except ValueError as err:
            logging.error("Testing encoder_helper: FAILED - %s", str(err))
            raise err

    def test_perform_feature_engineering(self) -> None:
        """
        Test the perform_feature_engineering function for data splitting.

        Verifies that the output shapes are correct and scaler is fitted.
        """
        try:
            test_df = self.data_frame.copy()
            test_df['Churn'] = test_df['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
            cat_columns = [
                'Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
            ]
            encoded_df = churn_library.encoder_helper(test_df, cat_columns, 'Churn')
            X_train, X_test, y_train, y_test, scaler = churn_library.perform_feature_engineering(
                encoded_df, 'Churn'
            )
            self.assertEqual(X_train.shape[0], y_train.shape[0])
            self.assertEqual(X_test.shape[0], y_test.shape[0])
            self.assertIsInstance(scaler, StandardScaler)
            self.assertGreater(len(X_train), 0)
            logging.info("Testing perform_feature_engineering: SUCCESS")
        except AssertionError as err:
            logging.error("Testing perform_feature_engineering: FAILED - %s", str(err))
            raise err
        except KeyError as err:
            logging.error("Testing perform_feature_engineering: FAILED - Missing columns: %s", str(err))
            raise err

    def test_classification_report_image(self) -> None:
        """
        Test the classification_report_image function for image generation.

        Verifies that plots are generated with valid inputs.
        """
        try:
            # Clean up existing plot files to ensure test accuracy
            for plot_file in [
                './image/logistic_classification_report.png',
                './image/random_forest_classification_report.png',
                './image/roc_curve.png'
            ]:
                if os.path.exists(plot_file):
                    os.remove(plot_file)

            # Valid input with consistent lengths
            y_train = np.array([0, 1, 0, 1])
            y_test = np.array([0, 1, 0, 1])
            y_train_preds_lr = np.array([0, 1, 0, 1])
            y_train_preds_rf = np.array([0, 1, 0, 1])
            y_test_preds_lr = np.array([0, 1, 0, 1])
            y_test_preds_rf = np.array([0, 1, 0, 1])
            y_test_probs_lr = np.array([0.1, 0.9, 0.5, 0.7])
            y_test_probs_rf = np.array([0.2, 0.8, 0.4, 0.6])

            churn_library.classification_report_image(
                y_train, y_test, y_train_preds_lr, y_train_preds_rf,
                y_test_preds_lr, y_test_preds_rf, y_test_probs_lr, y_test_probs_rf
            )
            self.assertTrue(os.path.exists('./image/logistic_classification_report.png'))
            self.assertTrue(os.path.exists('./image/random_forest_classification_report.png'))
            self.assertTrue(os.path.exists('./image/roc_curve.png'))
            logging.info("Testing classification_report_image: SUCCESS")
        except AssertionError as err:
            logging.error("Testing classification_report_image: FAILED - %s", str(err))
            raise err
        except Exception as err:
            logging.error("Testing classification_report_image: FAILED - %s", str(err))
            raise err

    def test_feature_importance_plot(self) -> None:
        """
        Test the feature_importance_plot function for plot generation.

        Uses a mock model to verify plot creation.
        """
        try:
            X_data = self.data_frame[[
                'Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon'
            ]].head(100)
            # Create Churn column for y_data
            y_data = self.data_frame['Attrition_Flag'].head(100).apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
            self.test_model.fit(X_data, y_data)
            churn_library.feature_importance_plot(self.test_model, X_data, './image/feature_importance.png')
            self.assertTrue(os.path.exists('./image/feature_importance.png'))
            self.assertTrue(os.path.exists('./image/shap_summary_plot.png'))
            logging.info("Testing feature_importance_plot: SUCCESS")
        except AssertionError as err:
            logging.error("Testing feature_importance_plot: FAILED - %s", str(err))
            raise err
        except KeyError as err:
            logging.error("Testing feature_importance_plot: FAILED - Missing columns: %s", str(err))
            raise err

    def test_train_models(self) -> None:
        """
        Test the train_models function for model training and output generation.

        Verifies that models and plots are saved correctly.
        """
        try:
            test_df = self.data_frame.copy()
            test_df['Churn'] = test_df['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
            cat_columns = [
                'Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
            ]
            encoded_df = churn_library.encoder_helper(test_df, cat_columns, 'Churn')
            X_train, X_test, y_train, y_test, scaler = churn_library.perform_feature_engineering(
                encoded_df, 'Churn'
            )
            churn_library.train_models(X_train, X_test, y_train, y_test, scaler)
            self.assertTrue(os.path.exists('./models/rfc_model.pkl'))
            self.assertTrue(os.path.exists('./models/logistic_model.pkl'))
            self.assertTrue(os.path.exists('./models/scaler.pkl'))
            self.assertTrue(os.path.exists('./image/roc_curve.png'))
            logging.info("Testing train_models: SUCCESS")
        except AssertionError as err:
            logging.error("Testing train_models: FAILED - %s", str(err))
            raise err
        except KeyError as err:
            logging.error("Testing train_models: FAILED - Missing columns: %s", str(err))
            raise err

if __name__ == '__main__':
    unittest.main()