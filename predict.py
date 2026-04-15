import numpy as np
from src.utils import load_model
import src.config as config

# 1. Load trained model
model = load_model(config.MODEL_PATH)

# 2. New input (age, physical_score)
sample = np.array([[40, 45]])

# 3. Predict
prediction = model.predict(sample)

print("Prediction:", prediction)
