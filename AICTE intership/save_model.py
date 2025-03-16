import pickle
from lifelines import CoxPHFitter
import pandas as pd

# Load your dataset (make sure the path is correct)
file_path = 'C://Users//prade//OneDrive//Desktop//AICTE intership//Dataset//heart.xlsx'
df = pd.read_excel(file_path)

# Ensure necessary columns are present
df['duration'] = df['time_to_progression'].clip(lower=0.1)
df['event'] = (df['Progression Score'] > df['Progression Score'].quantile(0.75)).astype(int)

# Define features
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal']

# Prepare data
X = df[['duration'] + features]
y = df['event']

# Create training DataFrame
train_df = pd.DataFrame(X, columns=X.columns)
train_df['event'] = y

# Train the Cox Proportional Hazards Model
cph_final = CoxPHFitter(penalizer=0.1)
cph_final.fit(train_df, duration_col='duration', event_col='event')

import pickle

model_file = "heart_disease_model.pkl"

# Save the trained Cox model
with open(model_file, "wb") as file:
    pickle.dump(cph_final, file)

print("Model saved successfully!")
