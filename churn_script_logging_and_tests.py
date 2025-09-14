import unittest
import logging
import os
import pandas as pd
import churn_library as cls

# Create logs directory and configure logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(
    filename='./logs/churn_script.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

class TestEDAPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures by loading the DataFrame."""
        try:
            self.df = cls.import_data("./data/bank_data.csv")
        except FileNotFoundError as err:
            logging.error("setUp: Failed to load DataFrame - File not found")
            raise err

    def test_import(self):
        """Test the import_data function."""
        try:
            df = cls.import_data("./data/bank_data.csv")
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_data: The file wasn't found")
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
            # Log DataFrame properties
            logging.info(f"DataFrame shape: {self.df.shape}")
            missing_values = self.df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                logging.info(f"Columns with missing values:\n{missing_values.to_string()}")
            else:
                logging.info("No columns with missing values")
            logging.info(f"DataFrame description:\n{self.df.describe().to_string()}")

            # Run the perform_eda function
            cls.perform_eda(self.df)
            logging.info("Testing perform_eda: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing perform_eda: Failed to execute - {str(err)}")
            raise err

        try:
            # Check if the Churn column was added
            self.assertIn('Churn', self.df.columns, "Churn column not found in DataFrame")
            self.assertTrue(self.df['Churn'].isin([0, 1]).all(), "Churn column contains invalid values")
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
        
    @unittest.skip("Not implemented yet")
    def test_encoder_helper(self):
        """Test the encoder_helper function."""
        try:
            # Assume encoder_helper encodes categorical columns into numerical ones
            category_lst = ['Marital_Status']  # Example categorical column
            response = 'Churn'  # Response variable
            encoded_df = cls.encoder_helper(self.df, category_lst, response)
            logging.info("Testing encoder_helper: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing encoder_helper: Failed to execute - {str(err)}")
            raise err

        try:
            # Check if encoded columns were added (e.g., Marital_Status_Churn)
            for col in category_lst:
                encoded_col = f"{col}_{response}"
                self.assertIn(encoded_col, encoded_df.columns, f"Encoded column {encoded_col} not found")
                self.assertTrue(encoded_df[encoded_col].notnull().all(), f"Encoded column {encoded_col} contains null values")
            logging.info("Testing encoder_helper: Encoded columns added successfully")
        except AssertionError as err:
            logging.error(f"Testing encoder_helper: Encoded column check failed - {str(err)}")
            raise err
	
    @unittest.skip("Not implemented yet")
    def test_perform_feature_engineering(self):
        """Test the perform_feature_engineering function."""
        try:
            # Assume perform_feature_engineering returns train/test split
            X_train, X_test, y_train, y_test = cls.perform_feature_engineering(self.df)
            logging.info("Testing perform_feature_engineering: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing perform_feature_engineering: Failed to execute - {str(err)}")
            raise err

        try:
            # Check if returned objects are non-empty
            self.assertGreater(X_train.shape[0], 0, "X_train has no rows")
            self.assertGreater(X_test.shape[0], 0, "X_test has no rows")
            self.assertGreater(len(y_train), 0, "y_train has no values")
            self.assertGreater(len(y_test), 0, "y_test has no values")
            # Check if train/test split sizes are reasonable (e.g., test size ~20%)
            total_rows = X_train.shape[0] + X_test.shape[0]
            self.assertAlmostEqual(total_rows, self.df.shape[0], msg="Train/test split does not match original DataFrame size")
            logging.info("Testing perform_feature_engineering: Train/test split sizes are valid")
        except AssertionError as err:
            logging.error(f"Testing perform_feature_engineering: Split check failed - {str(err)}")
            raise err
        
    @unittest.skip("Not implemented yet")
    def test_train_models(self):
        """Test the train_models function."""
        try:
            # Assume perform_feature_engineering returns train/test split
            X_train, X_test, y_train, y_test = cls.perform_feature_engineering(self.df)
            # Assume train_models trains models and saves them
            cls.train_models(X_train, X_test, y_train, y_test)
            logging.info("Testing train_models: SUCCESS - Function executed without errors")
        except Exception as err:
            logging.error(f"Testing train_models: Failed to execute - {str(err)}")
            raise err

        try:
            # Check if model output files exist (e.g., pickled models)
            expected_files = [
                './models/rfc_model.pkl',
                './models/logistic_model.pkl'
            ]  # Adjust based on actual model files
            for model_file in expected_files:
                self.assertTrue(os.path.isfile(model_file), f"Model file {model_file} not found")
            logging.info("Testing train_models: Model files created successfully")
        except AssertionError as err:
            logging.error(f"Testing train_models: Model file check failed - {str(err)}")
            raise err
        pass

if __name__ == '__main__':
    unittest.main()