import joblib
import pandas as pd
import numpy as np


rf = joblib.load('../models/model_rf.joblib')

scaler = joblib.load('../models/scaler.joblib')

new_data = pd.DataFrame(data=[[
2.4, 5.3, 0.56,12.8, 0.08, 3, 8, 0.9964, 4.15, 0.92, 6.7
]])

new_data_scaled = scaler.transform(new_data)

predictions = rf.predict(new_data_scaled)

probabilities = rf.predict_proba(new_data_scaled)

print('Predicted class: ', predictions)

print('Class probabilities: ', probabilities)
