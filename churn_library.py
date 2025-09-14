# library doc string


# import libraries
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Create image folder if it doesn't exist
    os.makedirs('image', exist_ok=True)

    # Basic EDA
    df.shape
    df.isnull().sum()
    df.describe()

    # Create Churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Create and save Churn histogram
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Churn Distribution')
    plt.savefig('image/churn_histogram.png')
    plt.close()

    # Create and save Customer Age histogram
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.savefig('image/customer_age_histogram.png')
    plt.close()

    # Create and save Marital Status bar plot
    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.savefig('image/marital_status_bar.png')
    plt.close()

    # Create and save Total Transaction Count histogram with KDE
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total Transaction Count Distribution')
    plt.savefig('image/total_trans_ct_histogram.png')
    plt.close()

    # Create and save correlation heatmap for numerical columns only
    plt.figure(figsize=(20, 10))
    # Select only numerical columns for correlation
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numerical_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig('image/correlation_heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


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
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
