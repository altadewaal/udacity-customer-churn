import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='./logs/churn_script.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

def import_data(pth):
    '''
    Returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        logging.info("import_data: SUCCESS - Loaded DataFrame from %s", pth)
        return df
    except FileNotFoundError as err:
        logging.error("import_data: File not found at %s - %s", pth, str(err))
        raise err
    except Exception as err:
        logging.error("import_data: Failed to load DataFrame - %s", str(err))
        raise err

def perform_eda(df):
    '''
    Perform EDA on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    try:
        if 'Churn' in df.columns:
            logging.error("perform_eda: Churn column already exists in DataFrame")
            raise ValueError("Churn column already exists in DataFrame")
        
        # Create Churn column
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        logging.info("perform_eda: Churn column created successfully")

        # Create images folder
        os.makedirs('./image', exist_ok=True)

        # Churn histogram
        plt.figure(figsize=(10, 6))
        df['Churn'].hist()
        plt.title('Churn Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.savefig('./image/churn_histogram.png')
        plt.close()
        logging.info("perform_eda: Saved churn_histogram.png")

        # Customer Age histogram
        plt.figure(figsize=(10, 6))
        df['Customer_Age'].hist()
        plt.title('Customer Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.savefig('./image/customer_age_histogram.png')
        plt.close()
        logging.info("perform_eda: Saved customer_age_histogram.png")

        # Marital Status bar plot
        plt.figure(figsize=(10, 6))
        df['Marital_Status'].value_counts().plot(kind='bar')
        plt.title('Marital Status Distribution')
        plt.xlabel('Marital Status')
        plt.ylabel('Count')
        plt.savefig('./image/marital_status_bar.png')
        plt.close()
        logging.info("perform_eda: Saved marital_status_bar.png")

        # Total Transaction Count histogram
        plt.figure(figsize=(10, 6))
        df['Total_Trans_Ct'].hist()
        plt.title('Total Transaction Count Distribution')
        plt.xlabel('Total Transaction Count')
        plt.ylabel('Count')
        plt.savefig('./image/total_trans_ct_histogram.png')
        plt.close()
        logging.info("perform_eda: Saved total_trans_ct_histogram.png")

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig('./image/correlation_heatmap.png')
        plt.close()
        logging.info("perform_eda: Saved correlation_heatmap.png")

    except Exception as err:
        logging.error("perform_eda: Failed to complete EDA - %s", str(err))
        raise err

def encoder_helper(df, category_lst, response='Churn'):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook
    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]
    output:
            df: pandas dataframe with new columns for each categorical variable containing the proportion of the response variable (e.g., Churn) for each category
    '''
    try:
        # Validate response column
        if response not in df.columns:
            logging.error(f"encoder_helper: Response column '{response}' not found in DataFrame")
            raise ValueError(f"Response column '{response}' not found in DataFrame")

        # Validate categorical columns
        for col in category_lst:
            if col not in df.columns:
                logging.error(f"encoder_helper: Categorical column '{col}' not found in DataFrame")
                raise ValueError(f"Categorical column '{col}' not found in DataFrame")

        # Create new columns with churn proportions
        for col in category_lst:
            # Calculate churn proportion for each category
            churn_rates = df.groupby(col)[response].mean()
            # Create new column name
            new_col = f"{col}_{response}"
            # Map categories to their churn proportions
            df[new_col] = df[col].map(churn_rates)
            logging.info(f"encoder_helper: Created column '{new_col}' with churn proportions")
        
        return df

    except Exception as err:
        logging.error(f"encoder_helper: Failed to process categorical columns - %s", str(err))
        raise err

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              scaler: fitted StandardScaler object
    '''
    try:
        keep_cols = [
            'Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
            'Income_Category_Churn', 'Card_Category_Churn'
        ]

        X = pd.DataFrame()
        X[keep_cols] = df[keep_cols]
        y = df[response]

        # Initialize and fit scaler on the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        logging.info("perform_feature_engineering: SUCCESS - Train-test split completed with scaled features")
        return X_train, X_test, y_train, y_test, scaler

    except Exception as err:
        logging.error(f"perform_feature_engineering: Failed to execute - %s", str(err))
        raise err

def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf, y_test_probs_lr, y_test_probs_rf):
    '''
    Produces classification report for training and testing results and stores report as image in images folder
    input:
            y_train: training response values
            y_test: test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            y_test_probs_lr: test probabilities from logistic regression
            y_test_probs_rf: test probabilities from random forest
    output:
             None
    '''
    try:
        # Create images folder
        os.makedirs('./image', exist_ok=True)

        # Logistic Regression Classification Report
        plt.figure(figsize=(8, 6))
        plt.text(0.01, 0.05, str('Logistic Regression Train\n' + classification_report(y_train, y_train_preds_lr)), 
                 {'fontsize': 10}, fontfamily='monospace')
        plt.text(0.01, 0.55, str('Logistic Regression Test\n' + classification_report(y_test, y_test_preds_lr)), 
                 {'fontsize': 10}, fontfamily='monospace')
        plt.axis('off')
        plt.savefig('./image/logistic_classification_report.png')
        plt.close()
        logging.info("classification_report_image: Saved Logistic Regression classification report to ./image/logistic_classification_report.png")

        # Random Forest Classification Report
        plt.figure(figsize=(8, 6))
        plt.text(0.01, 0.05, str('Random Forest Train\n' + classification_report(y_train, y_train_preds_rf)), 
                 {'fontsize': 10}, fontfamily='monospace')
        plt.text(0.01, 0.55, str('Random Forest Test\n' + classification_report(y_test, y_test_preds_rf)), 
                 {'fontsize': 10}, fontfamily='monospace')
        plt.axis('off')
        plt.savefig('./image/random_forest_classification_report.png')
        plt.close()
        logging.info("classification_report_image: Saved Random Forest classification report to ./image/random_forest_classification_report.png")

        # ROC Curve
        rfc_fpr, rfc_tpr, _ = roc_curve(y_test, y_test_probs_rf)
        lrc_fpr, lrc_tpr, _ = roc_curve(y_test, y_test_probs_lr)
        plt.figure(figsize=(10, 6))
        plt.plot(rfc_fpr, rfc_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_test_probs_rf):.2f})')
        plt.plot(lrc_fpr, lrc_tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_test_probs_lr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('./image/roc_curve.png')
        plt.close()
        logging.info("classification_report_image: Saved ROC curve to ./image/roc_curve.png")

    except Exception as err:
        logging.error(f"classification_report_image: Failed to generate or save images - %s", str(err))
        raise err

def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    try:
        # Get feature importances
        importances = model.feature_importances_
        feature_names = X_data.columns
        feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar')
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(output_pth)
        plt.close()
        logging.info("feature_importance_plot: Saved feature importance plot to %s", output_pth)

    except Exception as err:
        logging.error(f"feature_importance_plot: Failed to generate or save plot - %s", str(err))
        raise err

def train_models(X_train, X_test, y_train, y_test, scaler):
    '''
    Train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              scaler: fitted StandardScaler object
    output:
              None
    '''
    try:
        # Create models and images folders
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./image', exist_ok=True)

        # Initialize models
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=5000, random_state=42)

        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        # Grid search for Random Forest
        logging.info("train_models: Starting GridSearchCV for Random Forest")
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        logging.info("train_models: Completed GridSearchCV for Random Forest. Best params: %s", cv_rfc.best_params_)

        # Train Logistic Regression
        logging.info("train_models: Training Logistic Regression")
        lrc.fit(X_train, y_train)
        logging.info("train_models: Completed Logistic Regression training")

        # Save models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        logging.info("train_models: Saved Random Forest model to ./models/rfc_model.pkl")
        joblib.dump(lrc, './models/logistic_model.pkl')
        logging.info("train_models: Saved Logistic Regression model to ./models/logistic_model.pkl")
        joblib.dump(scaler, './models/scaler.pkl')
        logging.info("train_models: Saved StandardScaler to ./models/scaler.pkl")

        # Generate predictions for classification reports
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        # Generate probabilities for ROC curve
        y_test_probs_lr = lrc.predict_proba(X_test)[:, 1]
        y_test_probs_rf = cv_rfc.best_estimator_.predict_proba(X_test)[:, 1]

        # Generate classification reports and ROC curve
        classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
            y_test_probs_lr,
            y_test_probs_rf
        )
        logging.info("train_models: Generated classification reports and ROC curve")

        # Generate feature importance plot for Random Forest
        feature_importance_plot(cv_rfc.best_estimator_, X_train, './image/feature_importance.png')
        logging.info("train_models: Generated feature importance plot")

    except Exception as err:
        logging.error(f"train_models: Failed to train or save models - %s", str(err))
        raise err

if __name__ == '__main__':
    try:
        # Get data file path from environment variable
        data_path = os.environ.get('BANK_DATA_PATH', './data/bank_data.csv')
        logging.info("main: Starting pipeline with data path: %s", data_path)

        # Import data
        df = import_data(data_path)
        logging.info("main: Data imported successfully")

        # Perform EDA
        perform_eda(df)
        logging.info("main: EDA completed")

        # Define categorical columns
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        # Encode categorical columns
        df_encoded = encoder_helper(df, cat_columns, response='Churn')
        logging.info("main: Categorical encoding completed")

        # Check RUN_FEATURE_ENGINEERING environment variable
        run_feature_engineering = os.environ.get('RUN_FEATURE_ENGINEERING', 'true').lower() == 'true'
        if run_feature_engineering:
            # Perform feature engineering
            X_train, X_test, y_train, y_test, scaler = perform_feature_engineering(df_encoded, response='Churn')
            logging.info("main: Feature engineering completed")

            # Train models
            train_models(X_train, X_test, y_train, y_test, scaler)
            logging.info("main: Model training and image generation completed")
        else:
            logging.info("main: Skipped feature engineering and model training (RUN_FEATURE_ENGINEERING=false)")

    except Exception as err:
        logging.error("main: Pipeline failed - %s", str(err))
        raise err