from src.preprocessing import load_and_preprocess_data
from src.model import train_and_save_model
from sklearn.metrics import accuracy_score
from src.model import train_and_save_model
# Load data
X, y = load_and_preprocess_data("data/telco.csv")

# Train model
model, X_test, y_test = train_and_save_model(X, y)

# Evaluate
y_pred = model.predict(X_test)

print("Model trained successfully!")
print("Accuracy:", accuracy_score(y_test, y_pred))