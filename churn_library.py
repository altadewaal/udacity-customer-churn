"""
Churn Prediction Pipeline

This module provides a machine learning pipeline for predicting customer churn from bank data.
It includes functions to import data, perform exploratory data analysis (EDA), encode categorical
features, engineer features, train models, and generate evaluation metrics and visualizations.
Outputs include saved models in './models/' and plots in './image/'.

Functions:
    import_data: Loads a CSV file into a pandas DataFrame.
    perform_eda: Generates EDA visualizations and adds a 'Churn' column.
    encoder_helper: Encodes categorical columns with churn proportions.
    perform_feature_engineering: Splits data and scales features for modeling.
    classification_report_image: Creates classification reports and ROC curve plots.
    feature_importance_plot: Generates feature importance and SHAP plots.
    train_models: Trains Logistic Regression and Random Forest models.
    save_model: Saves a trained model to disk.
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

def import_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV dataset into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        Exception: For other loading errors.
    """
    try:
        data_frame = pd.read_csv(file_path)
        return data_frame
    except FileNotFoundError as err:
        raise FileNotFoundError(f"File not found: {file_path}") from err
    except Exception as err:
        raise Exception(f"Error loading data: {str(err)}") from err

def perform_eda(data_frame: pd.DataFrame) -> None:
    """
    Performs exploratory data analysis and saves visualizations to './image/'.

    Args:
        data_frame (pd.DataFrame): Input DataFrame.

    Raises:
        ValueError: If 'Churn' column already exists.
        Exception: For other EDA errors.
    """
    try:
        if 'Churn' in data_frame.columns:
            raise ValueError("Churn column already exists in DataFrame")

        # Create Churn column
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        # Create images folder
        os.makedirs('./image', exist_ok=True)

        # Churn histogram
        plt.figure(figsize=(10, 6))
        data_frame['Churn'].hist()
        plt.title('Churn Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.savefig('./image/churn_histogram.png')
        plt.close()

        # Customer Age histogram
        plt.figure(figsize=(10, 6))
        data_frame['Customer_Age'].hist()
        plt.title('Customer Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.savefig('./image/customer_age_histogram.png')
        plt.close()

        # Marital Status bar plot
        plt.figure(figsize=(10, 6))
        data_frame['Marital_Status'].value_counts().plot(kind='bar')
        plt.title('Marital Status Distribution')
        plt.xlabel('Marital Status')
        plt.ylabel('Count')
        plt.savefig('./image/marital_status_bar.png')
        plt.close()

        # Total Transaction Count histogram
        plt.figure(figsize=(10, 6))
        data_frame['Total_Trans_Ct'].hist()
        plt.title('Total Transaction Count Distribution')
        plt.xlabel('Total Transaction Count')
        plt.ylabel('Count')
        plt.savefig('./image/total_trans_ct_histogram.png')
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            data_frame.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm',
            fmt='.2f'
        )
        plt.title('Correlation Heatmap')
        plt.savefig('./image/correlation_heatmap.png')
        plt.close()

    except Exception as err:
        raise Exception(f"EDA failed: {str(err)}") from err

def encoder_helper(
    data_frame: pd.DataFrame,
    category_list: list[str],
    response: str = 'Churn'
) -> pd.DataFrame:
    """
    Encodes categorical columns with churn proportions.

    Args:
        data_frame (pd.DataFrame): Input DataFrame.
        category_list (list[str]): List of categorical column names.
        response (str): Response column name (default: 'Churn').

    Returns:
        pd.DataFrame: DataFrame with new encoded columns.

    Raises:
        ValueError: If response or categorical columns are missing.
        Exception: For other encoding errors.
    """
    try:
        # Validate response column
        if response not in data_frame.columns:
            raise ValueError(f"Response column '{response}' not found in DataFrame")

        # Validate categorical columns
        missing_cols = [col for col in category_list if col not in data_frame.columns]
        if missing_cols:
            raise ValueError(f"Categorical columns not found in DataFrame: {missing_cols}")

        result_df = data_frame.copy()
        for col in category_list:
            churn_rates = result_df.groupby(col)[response].mean()
            new_col = f"{col}_{response}"
            result_df[new_col] = result_df[col].map(churn_rates)

        return result_df

    except Exception as err:
        raise Exception(f"Encoding failed: {str(err)}") from err

def perform_feature_engineering(
    data_frame: pd.DataFrame,
    response: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Prepares features and splits data for modeling.

    Args:
        data_frame (pd.DataFrame): Input DataFrame with encoded features.
        response (str): Response column name.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler) containing training and
               test data and the fitted scaler.

    Raises:
        KeyError: If required columns are missing.
        Exception: For other feature engineering errors.
    """
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

        # Validate required columns
        missing_cols = [col for col in keep_cols if col not in data_frame.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        X = data_frame[keep_cols].copy()
        y = data_frame[response]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train_scaled, columns=keep_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=keep_cols, index=X_test.index)

        return X_train, X_test, y_train, y_test, scaler

    except KeyError as err:
        raise KeyError(f"Feature engineering failed: {str(err)}") from err
    except Exception as err:
        raise Exception(f"Feature engineering failed: {str(err)}") from err

def classification_report_image(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_preds_lr: np.ndarray,
    y_train_preds_rf: np.ndarray,
    y_test_preds_lr: np.ndarray,
    y_test_preds_rf: np.ndarray,
    y_test_probs_lr: np.ndarray,
    y_test_probs_rf: np.ndarray
) -> None:
    """
    Generates classification reports and ROC curve plots, saving to './image/'.

    Args:
        y_train (np.ndarray): Training response values.
        y_test (np.ndarray): Test response values.
        y_train_preds_lr (np.ndarray): Training predictions from logistic regression.
        y_train_preds_rf (np.ndarray): Training predictions from random forest.
        y_test_preds_lr (np.ndarray): Test predictions from logistic regression.
        y_test_preds_rf (np.ndarray): Test predictions from random forest.
        y_test_probs_lr (np.ndarray): Test probabilities from logistic regression.
        y_test_probs_rf (np.ndarray): Test probabilities from random forest.

    Raises:
        ValueError: If input lengths mismatch or contain invalid values.
        Exception: For other plotting errors.
    """
    try:
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train_preds_lr = np.array(y_train_preds_lr)
        y_train_preds_rf = np.array(y_train_preds_rf)
        y_test_preds_lr = np.array(y_test_preds_lr)
        y_test_preds_rf = np.array(y_test_preds_rf)
        y_test_probs_lr = np.array(y_test_probs_lr)
        y_test_probs_rf = np.array(y_test_probs_rf)

        n_test = len(y_test)
        if not (len(y_test_probs_lr) == n_test and len(y_test_probs_rf) == n_test):
            raise ValueError("Mismatched lengths between y_test and probability arrays")

        if (np.any(np.isnan(y_test)) or np.any(np.isnan(y_test_probs_lr)) or
                np.any(np.isnan(y_test_probs_rf)) or np.any(np.isinf(y_test)) or
                np.any(np.isinf(y_test_probs_lr)) or np.any(np.isinf(y_test_probs_rf))):
            raise ValueError("Input contains NaN or infinite values")

        if not np.all(np.isin(y_test, [0, 1])):
            raise ValueError("y_test must contain only binary values (0 or 1)")

        if not (np.all(y_test_probs_lr >= 0) and np.all(y_test_probs_lr <= 1) and
                np.all(y_test_probs_rf >= 0) and np.all(y_test_probs_rf <= 1)):
            raise ValueError("Probability values must be between 0 and 1")

        os.makedirs('./image', exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.text(
            0.01, 0.05,
            f"Logistic Regression Train\n{classification_report(y_train, y_train_preds_lr)}",
            {'fontsize': 10}, fontfamily='monospace'
        )
        plt.text(
            0.01, 0.55,
            f"Logistic Regression Test\n{classification_report(y_test, y_test_preds_lr)}",
            {'fontsize': 10}, fontfamily='monospace'
        )
        plt.axis('off')
        plt.savefig('./image/logistic_classification_report.png')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.text(
            0.01, 0.05,
            f"Random Forest Train\n{classification_report(y_train, y_train_preds_rf)}",
            {'fontsize': 10}, fontfamily='monospace'
        )
        plt.text(
            0.01, 0.55,
            f"Random Forest Test\n{classification_report(y_test, y_test_preds_rf)}",
            {'fontsize': 10}, fontfamily='monospace'
        )
        plt.axis('off')
        plt.savefig('./image/random_forest_classification_report.png')
        plt.close()

        rfc_fpr, rfc_tpr, _ = roc_curve(y_test, y_test_probs_rf)
        lrc_fpr, lrc_tpr, _ = roc_curve(y_test, y_test_probs_lr)
        plt.figure(figsize=(10, 6))
        plt.plot(
            rfc_fpr, rfc_tpr,
            label=f"Random Forest (AUC = {roc_auc_score(y_test, y_test_probs_rf):.2f})"
        )
        plt.plot(
            lrc_fpr, lrc_tpr,
            label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_test_probs_lr):.2f})"
        )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('./image/roc_curve.png')
        plt.close()

    except Exception as err:
        raise Exception(f"Classification report plotting failed: {str(err)}") from err

def feature_importance_plot(model: RandomForestClassifier, X_data: pd.DataFrame, output_path: str) -> None:
    """
    Creates and saves feature importance and SHAP summary plots.

    Args:
        model (RandomForestClassifier): Trained model with feature_importances_.
        X_data (pd.DataFrame): Feature DataFrame.
        output_path (str): Path to save the feature importance plot.

    Raises:
        AttributeError: If model lacks feature_importances_.
        Exception: For other plotting errors.
    """
    try:
        os.makedirs('./image', exist_ok=True)

        importances = model.feature_importances_
        feature_names = X_data.columns
        feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar')
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        shap_output_path = os.path.join(os.path.dirname(output_path), 'shap_summary_plot.png')
        plt.savefig(shap_output_path)
        plt.close()

    except Exception as err:
        raise Exception(f"Feature importance plotting failed: {str(err)}") from err

def save_model(model: object, file_path: str) -> None:
    """
    Saves a trained model to disk.

    Args:
        model (object): Trained model to save.
        file_path (str): Path to save the model.

    Raises:
        Exception: For saving errors.
    """
    try:
        joblib.dump(model, file_path)
    except Exception as err:
        raise Exception(f"Model saving failed: {str(err)}") from err

def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler
) -> None:
    """
    Trains Logistic Regression and Random Forest models, saves results and models.

    Args:
        X_train (pd.DataFrame): Training feature data.
        X_test (pd.DataFrame): Test feature data.
        y_train (pd.Series): Training response data.
        y_test (pd.Series): Test response data.
        scaler (StandardScaler): Fitted scaler object.

    Raises:
        Exception: For training or saving errors.
    """
    try:
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./image', exist_ok=True)

        # Train Random Forest with GridSearchCV
        rfc = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        # Train Logistic Regression
        lrc = LogisticRegression(solver='lbfgs', max_iter=5000, random_state=42)
        lrc.fit(X_train, y_train)

        # Save models
        save_model(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        save_model(lrc, './models/logistic_model.pkl')
        save_model(scaler, './models/scaler.pkl')

        # Generate predictions
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
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

        # Generate feature importance plot
        feature_importance_plot(cv_rfc.best_estimator_, X_test, './image/feature_importance.png')

    except Exception as err:
        raise Exception(f"Model training failed: {str(err)}") from err

if __name__ == '__main__':
    try:
        load_dotenv()
        data_path = os.environ.get('BANK_DATA_PATH', './data/bank_data.csv')
        data_frame = import_data(data_path)
        perform_eda(data_frame)
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        encoded_df = encoder_helper(data_frame, cat_columns, response='Churn')
        run_feature_engineering = os.environ.get('RUN_FEATURE_ENGINEERING', 'true').lower() == 'true'
        if run_feature_engineering:
            X_train, X_test, y_train, y_test, scaler = perform_feature_engineering(
                encoded_df, response='Churn'
            )
            train_models(X_train, X_test, y_train, y_test, scaler)
    except Exception as err:
        raise err