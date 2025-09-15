import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, RocCurveDisplay


# Configure logging
logging.basicConfig(
    filename='./logs/churn_script.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

# Load .env file
load_dotenv()
DATA_FILE_PATH = os.environ.get('BANK_DATA_PATH', './data/bank_data.csv')

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

    # Check if Churn column already exists
    if 'Churn' in df.columns:
        raise ValueError("Churn column already exists in DataFrame")

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

    return df


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
        logging.error(f"encoder_helper: Failed to process categorical columns - {str(err)}")
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
    '''
    

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    y = df[response]

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

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
    df = import_data(DATA_FILE_PATH)
    df_Churn = perform_eda(df)

    #Categorical columns
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

    df_Cat = encoder_helper(df_Churn, cat_columns, response='Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(df_Cat, 
                                                                   response='Churn')
    
