from src import load_data, build_model, train, evaluate, save_model, config

# 1. Load data
X_train, X_test, y_train, y_test = load_data(
    config.DATA_PATH,
    config.TEST_SIZE,
    config.RANDOM_STATE
)

# 2. Build model
model = build_model(config.MAX_ITER)

# 3. Train model
model = train(model, X_train, y_train)

# 4. Evaluate model
acc, report = evaluate(model, X_test, y_test)

print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", report)

# 5. Save model
save_model(model, config.MODEL_PATH)

print("\nModel saved successfully!")
