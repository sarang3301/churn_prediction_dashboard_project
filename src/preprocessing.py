import pandas as pd
import numpy as np

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop ID
    df.drop('customerID', axis=1, inplace=True)

    # Split X, y
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Handle missing values
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    cat_cols = X.select_dtypes(include=['object']).columns
    X[cat_cols] = X[cat_cols].fillna('Unknown')

    # Encoding
    X = pd.get_dummies(X, drop_first=True)

    # Final clean
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y