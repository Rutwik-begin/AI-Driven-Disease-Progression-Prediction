import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import KaplanMeierFitter, CoxPHFitter

# --- 1. Data Loading and Initial Overview ---

# Step 1: Importing the data
file_path = ('C://Users//prade//OneDrive//Desktop//AICTE intership//Dataset//heart.xlsx')
df = pd.read_excel(file_path)

# Step 2: Overview of the dataset
print("--- Dataset Head ---")
print(df.head())

# --- 2. Data Preprocessing and Feature Engineering ---

# Step 3: Define high-risk threshold using 75th percentile of Progression Score
threshold = df['Progression Score'].quantile(0.75)

# Step 4: Create a binary event column (1 if above threshold, 0 otherwise)
df['event'] = (df['Progression Score'] > threshold).astype(int)

# Step 5: Count the number of events
print("\n--- Event Counts ---")
print(df['event'].value_counts())

# Step 6: Verify Progression Score and event mapping
print("\n--- Progression Score vs. Event Mapping (Head) ---")
print(df[['Progression Score', 'event']].head())

# Step 7: Standardize selected features
features_to_scale = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                    'restecg', 'thalach', 'exang', 'oldpeak',
                    'slope', 'ca', 'thal']
scaler = StandardScaler()
df[features_to_scale] = pd.DataFrame(scaler.fit_transform(df[features_to_scale]),
                                    columns=features_to_scale,
                                    index=df.index)

# Define the initial features list
features = [col for col in df.columns if col not in ['Progression Score', 'event', 'duration', 'target', 'time_to_progression']]

# Step 11: Compute Variance Inflation Factor (VIF) and drop high-VIF features
X_vif = df[features]
vif_data = pd.DataFrame()
vif_data["Feature"] = features
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
if high_vif_features:
    df = df.drop(columns=high_vif_features)
    features = [col for col in features if col not in high_vif_features]  # Update features list

# Step 15: Check for near-constant columns and drop them
near_constant_cols = [col for col in df.columns if df[col].std() < 1e-3]
if near_constant_cols:
    df = df.drop(columns=near_constant_cols)
    features = [col for col in features if col not in near_constant_cols]  # Update features list

# Step 13 & 14: Create 'duration' from 'time_to_progression'
if 'time_to_progression' in df.columns:
    df['duration'] = df['time_to_progression'].clip(lower=0.1)
    df = df.drop(columns=['time_to_progression'])

# Step 17: Split dataset into train and test sets
X = df[['duration'] + features]  # Use the updated features list, without event
y = df['event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure train_df and test_df remain DataFrames
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['event'] = y_train.values

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['event'] = y_test.values

print("\n--- Final Features List Used for Cox Model ---")
print(features)

# --- 3. Exploratory Data Analysis (EDA) ---

# Step 18: Plot histogram of Progression Score distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Progression Score'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Progression Scores')
plt.xlabel('Progression Score')
plt.ylabel('Frequency')
plt.show()

# Step 19: Plot correlation heatmap to identify relationships between features
plt.figure(figsize=(10, 8))
corr_matrix = df[features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# --- 4. Survival Analysis: Kaplan-Meier and Cox PH Model ---

# Step 20: Kaplan-Meier Survival Analysis
median_progression_score = df['Progression Score'].median()
high_risk = df[df['Progression Score'] > median_progression_score]
low_risk = df[df['Progression Score'] <= median_progression_score]

cph = CoxPHFitter(penalizer=0.1)
cph.fit(train_df, duration_col='duration', event_col='event')

kmf = KaplanMeierFitter()

plt.figure(figsize=(12, 8))
kmf.fit(high_risk['duration'], high_risk['event'], label='High Risk')
kmf.plot_survival_function(ci_show=True, ci_alpha=0.2)
kmf.fit(low_risk['duration'], low_risk['event'], label='Low Risk')
kmf.plot_survival_function(ci_show=True, ci_alpha=0.2)
plt.title('Kaplan-Meier Survival Curves')
plt.xlabel('Time to Progression (Months)')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)
plt.show()

# Cross-validation for Cox Proportional Hazards model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_indices = []

for fold, (train_index, test_index) in enumerate(kf.split(df)):
    train_data = df.iloc[train_index].copy()
    test_data = df.iloc[test_index].copy()

    try:
        cph.fit(train_data, duration_col='duration', event_col='event')
        c_index = cph.score(test_data, scoring_method='concordance_index')
        c_indices.append(c_index)
    except Exception as e:
        print(f"Error in fold {fold + 1}: {type(e).__name__} - {e}")
        c_indices.append(None)

c_indices = [ci for ci in c_indices if ci is not None]

print("\n--- Cross-Validation C-Indices ---")
print("Cross-Validation C-Indices:", c_indices)
print("Average C-Index:", np.mean(c_indices) if c_indices else "N/A")

# Step 21: Cox Proportional Hazards Model training and validation
cph_final = CoxPHFitter(penalizer=0.1)
cph_final.fit(train_df, duration_col='duration', event_col='event')

c_index = cph_final.score(test_df, scoring_method='concordance_index')
print(f"\n--- Cox Model Performance ---")
print(f"Concordance Index on Test Set: {c_index:.3f}")

# Step 22: Checking proportional hazards assumption
print("\n--- Checking Proportional Hazards Assumptions ---")
cph_final.check_assumptions(train_df.loc[cph_final.event_observed.index], p_value_threshold=0.05)

# Calculate baseline hazard
baseline_hazard = cph_final.baseline_hazard_

# Step 23: Baseline hazard function plotting
plt.figure(figsize=(10, 6))
plt.plot(baseline_hazard.index, baseline_hazard['baseline hazard'])
plt.title('Baseline Hazard Function')
plt.xlabel('Time to Progression (Months)')
plt.ylabel('Hazard')
plt.grid(True)
plt.show()

# Bin 'chol', 'trestbps', and 'oldpeak' into 4 categories each
train_df['chol_bins'] = pd.cut(train_df['chol'], bins=4, labels=False)
train_df['trestbps_bins'] = pd.cut(train_df['trestbps'], bins=4, labels=False)
train_df['oldpeak_bins'] = pd.cut(train_df['oldpeak'], bins=4, labels=False)

# Fit the Cox model with stratification
cph.fit(train_df, duration_col='duration', event_col='event', strata=['restecg', 'chol_bins', 'fbs', 'trestbps_bins', 'oldpeak_bins'])

# Check proportional hazards assumptions again
cph.check_assumptions(train_df, p_value_threshold=0.05, show_plots=True)
plt.show()

# 1. Visualizing Feature Coefficients
plt.figure(figsize=(10, 6))
coefficients = cph.summary[['coef']].sort_values(by='coef')
sns.barplot(x=coefficients['coef'], y=coefficients.index, palette='coolwarm')
plt.title('Feature Coefficients from Cox Proportional Hazards Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.axvline(x=0, color='gray', linestyle='--')
plt.show()

# 2. Calculating Personalized Risk Scores for Sample Patients
# Extract sample rows from the test set
sample_patients = test_df.sample(5, random_state=42)
predicted_risks = cph.predict_partial_hazard(sample_patients)

# Display patient data alongside their risk scores
sample_patients['Predicted Risk Score'] = predicted_risks.values
print("Sample Patients with Predicted Risk Scores:")
print(sample_patients[['duration', 'event', 'Predicted Risk Score']])