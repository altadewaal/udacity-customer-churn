"""
Churn Prediction Pipeline

This module implements a machine learning pipeline for predicting customer churn based on bank customer data. It includes functions for data import, exploratory data analysis (EDA), feature engineering, model training, and evaluation. The pipeline processes a dataset to create a 'Churn' column, performs EDA to generate visualizations, encodes categorical features, trains Logistic Regression and Random Forest models, and produces evaluation metrics and plots, including feature importance and SHAP summary plots. Outputs are saved as model files and images.

When run as a script, it executes the full pipeline.

Functions:
- import_data: Loads a CSV dataset into a pandas DataFrame.
- perform_eda: Performs EDA and generates visualizations.
- encoder_helper: Encodes categorical columns with churn proportions.
- perform_feature_engineering: Prepares features and splits data for modeling.
- classification_report_image: Generates classification reports and ROC curves.
- feature_importance_plot: Creates feature importance and SHAP plots.
- train_models: Trains models and generates evaluation outputs.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv

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
        return df
    except FileNotFoundError as err:
        raise err
    except Exception as err:
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
            raise ValueError("Churn column already exists in DataFrame")
        
        # Create Churn column
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

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

        # Customer Age histogram
        plt.figure(figsize=(10, 6))
        df['Customer_Age'].hist()
        plt.title('Customer Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.savefig('./image/customer_age_histogram.png')
        plt.close()

        # Marital Status bar plot
        plt.figure(figsize=(10, 6))
        df['Marital_Status'].value_counts().plot(kind='bar')
        plt.title('Marital Status Distribution')
        plt.xlabel('Marital Status')
        plt.ylabel('Count')
        plt.savefig('./image/marital_status_bar.png')
        plt.close()

        # Total Transaction Count histogram
        plt.figure(figsize=(10, 6))
        df['Total_Trans_Ct'].hist()
        plt.title('Total Transaction Count Distribution')
        plt.xlabel('Total Transaction Count')
        plt.ylabel('Count')
        plt.savefig('./image/total_trans_ct_histogram.png')
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig('./image/correlation_heatmap.png')
        plt.close()

    except Exception as err:
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
            raise ValueError(f"Response column '{response}' not found in DataFrame")

        # Validate categorical columns
        for col in category_lst:
            if col not in df.columns:
                raise ValueError(f"Categorical column '{col}' not found in DataFrame")

        # Create new columns with churn proportions
        for col in category_lst:
            # Calculate churn proportion for each category
            churn_rates = df.groupby(col)[response].mean()
            # Create new column name
            new_col = f"{col}_{response}"
            # Map categories to their churn proportions
            df[new_col] = df[col].map(churn_rates)
        
        return df

    except Exception as err:
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

        X = df[keep_cols].copy()
        y = df[response]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Initialize and fit scaler on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train_scaled, columns=keep_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=keep_cols, index=X_test.index)

        return X_train, X_test, y_train, y_test, scaler

    except Exception as err:
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
        # Convert inputs to NumPy arrays for consistency
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train_preds_lr = np.array(y_train_preds_lr)
        y_train_preds_rf = np.array(y_train_preds_rf)
        y_test_preds_lr = np.array(y_test_preds_lr)
        y_test_preds_rf = np.array(y_test_preds_rf)
        y_test_probs_lr = np.array(y_test_probs_lr)
        y_test_probs_rf = np.array(y_test_probs_rf)

        # Validate input lengths
        n_test = len(y_test)
        if not (len(y_test_probs_lr) == n_test and len(y_test_probs_rf) == n_test):
            raise ValueError("Mismatched lengths between y_test and probability arrays")
        
        # Validate for NaN or infinite values
        if (np.any(np.isnan(y_test)) or np.any(np.isnan(y_test_probs_lr)) or np.any(np.isnan(y_test_probs_rf)) or
            np.any(np.isinf(y_test)) or np.any(np.isinf(y_test_probs_lr)) or np.any(np.isinf(y_test_probs_rf))):
            raise ValueError("Input contains NaN or infinite values")

        # Validate y_test is binary
        if not np.all(np.isin(y_test, [0, 1])):
            raise ValueError("y_test must contain only binary values (0 or 1)")

        # Validate probability ranges
        if not (np.all(y_test_probs_lr >= 0) and np.all(y_test_probs_lr <= 1) and
                np.all(y_test_probs_rf >= 0) and np.all(y_test_probs_rf <= 1)):
            raise ValueError("Probability values must be between 0 and 1")

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

        # Random Forest Classification Report
        plt.figure(figsize=(8, 6))
        plt.text(0.01, 0.05, str('Random Forest Train\n' + classification_report(y_train, y_train_preds_rf)), 
                 {'fontsize': 10}, fontfamily='monospace')
        plt.text(0.01, 0.55, str('Random Forest Test\n' + classification_report(y_test, y_test_preds_rf)), 
                 {'fontsize': 10}, fontfamily='monospace')
        plt.axis('off')
        plt.savefig('./image/random_forest_classification_report.png')
        plt.close()

        # ROC Curve
        rfc_fpr, rfc_tpr, _ = roc_curve(y_test, y_test_probs_rf)
        lrc_fpr, lrc_tpr, _ = roc_curve(y_test, y_test_probs_lr)
        plt.figure(figsize=(10, 6))
        plt.plot(rfc_fpr, rfc_tpr, label='Random Forest (AUC = {:.2f})'.format(roc_auc_score(y_test, y_test_probs_rf)))
        plt.plot(lrc_fpr, lrc_tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(y_test, y_test_probs_lr)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('./image/roc_curve.png')
        plt.close()

    except Exception as err:
        raise err

def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances and SHAP summary plot in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    try:
        # Create images folder
        os.makedirs('./image', exist_ok=True)

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

        # Generate SHAP summary plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        shap_output_pth = os.path.join(os.path.dirname(output_pth), 'shap_summary_plot.png')
        plt.savefig(shap_output_pth)
        plt.close()

    except Exception as err:
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
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        # Train Logistic Regression
        lrc.fit(X_train, y_train)

        # Save models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        joblib.dump(scaler, './models/scaler.pkl')

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

        # Generate feature importance plot for Random Forest
        feature_importance_plot(cv_rfc.best_estimator_, X_test, './image/feature_importance.png')

    except Exception as err:
        raise err

if __name__ == '__main__':
    try:
        # Load environment variables
        load_dotenv()
        data_path = os.environ.get('BANK_DATA_PATH', './data/bank_data.csv')

        # Import data
        df = import_data(data_path)

        # Perform EDA
        perform_eda(df)

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

        # Check RUN_FEATURE_ENGINEERING environment variable
        run_feature_engineering = os.environ.get('RUN_FEATURE_ENGINEERING', 'true').lower() == 'true'
        if run_feature_engineering:
            # Perform feature engineering
            X_train, X_test, y_train, y_test, scaler = perform_feature_engineering(df_encoded, response='Churn')

            # Train models
            train_models(X_train, X_test, y_train, y_test, scaler)
    except Exception as err:
        raise err