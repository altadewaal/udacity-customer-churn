import os
import logging
import churn_library as cls

os.makedirs('./logs', exist_ok=True)
log_file = "./logs/churn_script.log"
log_dir = os.path.dirname(log_file)


logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function
    input:
                    df: pandas dataframe
    output:
                    None
    '''
    # Log DataFrame properties
    logging.info(f"DataFrame shape: {df.shape}")
    # Log only columns with missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        logging.info(
            f"Columns with missing values:\n{
                missing_values.to_string()}")
    else:
        logging.info("No columns with missing values")
    logging.info(f"DataFrame description:\n{df.describe().to_string()}")

    # Test if the function runs without error
    try:
        # Run the perform_eda function
        perform_eda(df)
        logging.info(
            "Testing perform_eda: SUCCESS - Function executed without errors")
    except Exception as err:
        logging.error(f"Testing perform_eda: Failed to execute - {str(err)}")
        raise err

    try:
        # Check if the Churn column was added
        assert 'Churn' in df.columns, "Churn column not found in DataFrame"
        assert df['Churn'].isin([0, 1]).all(
        ), "Churn column contains invalid values"
        logging.info("Testing perform_eda: Churn column added successfully")
    except AssertionError as err:
        logging.error(
            f"Testing perform_eda: Churn column check failed - {str(err)}")
        raise err

    try:
        # Check if the image folder was created
        assert os.path.exists('image'), "Image folder was not created"
        logging.info("Testing perform_eda: Image folder created successfully")
    except AssertionError as err:
        logging.error("Testing perform_eda: Image folder creation failed")
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
            assert os.path.isfile(plot_path), f"Plot file {plot} not found"
        logging.info(
            "Testing perform_eda: All expected plot files created successfully")
    except AssertionError as err:
        logging.error(
            f"Testing perform_eda: Plot file check failed - {str(err)}")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    test_import(cls.import_data)
    df = cls.import_data("./data/bank_data.csv")
    test_eda(cls.perform_eda, df)
