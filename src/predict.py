import pickle
import pandas as pd
import json

def load_model():
    with open("models/model.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    with open("models/columns.json", "r") as f:
        columns = json.load(f)

    return model, scaler, columns


def predict(input_data):
    model, scaler, columns = load_model()

    # Create full dataframe
    df = pd.DataFrame(0, index=[0], columns=columns)

    # Fill input values safely
    for col in input_data.columns:
        if col in df.columns:
            df[col] = input_data[col]

    # Scale
    df_scaled = scaler.transform(df)

    return model.predict(df_scaled)