import os
os.makedirs("models", exist_ok=True)
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_and_save_model(X, y):
    # Split
       # 🔥 SAVE FEATURE COLUMNS HERE (IMPORTANT)
    with open("./models/columns.json", "w") as f:
        json.dump(list(X.columns), f)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model + scaler
    with open("./models/model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)

    return model, X_test, y_test