import config
from data import load_data
from model import get_model
from train import train_model
from utils import evaluate_model

# Load data
X_train, X_test, y_train, y_test = load_data(
    config.TEST_SIZE,
    config.RANDOM_STATE
)

# Model
model = get_model(config.MAX_ITER)

# Train
model = train_model(model, X_train, y_train)

# Evaluate
acc, cm, report = evaluate_model(model, X_test, y_test)

print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
