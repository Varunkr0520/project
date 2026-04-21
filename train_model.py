import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Get project root folder
base_dir = os.path.dirname(os.path.dirname(__file__))

# Correct dataset path
data_path = os.path.join(base_dir, "data_core.csv")

# Load dataset
data = pd.read_csv(data_path)

print("✅ Dataset Loaded")

# Clean
data.columns = data.columns.str.strip()
data = data.dropna()

# ----------------------------
# ✅ FIX 1: Standardize column names
# ----------------------------
data.columns = [col.lower().replace(" ", "_") for col in data.columns]

# ----------------------------
# ✅ FIX 2: Separate target FIRST
# ----------------------------
target_col = data.columns[-1]   # last column = crop
X = data.drop(columns=[target_col])
y = data[target_col]

# ----------------------------
# ✅ FIX 3: Encode ONLY feature columns
# ----------------------------
encoders = {}

for col in X.columns:
    try:
        X[col] = pd.to_numeric(X[col])
    except:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

# ----------------------------
# ✅ FIX 4: Encode target separately
# ----------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

encoders["target"] = target_encoder   # save target encoder

# ----------------------------
# Train
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ----------------------------
# Save model
# ----------------------------
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "crop_model.pkl")
joblib.dump(model, model_path)

# Save encoders
enc_path = os.path.join(model_dir, "encoders.pkl")
joblib.dump(encoders, enc_path)

print("✅ Model saved at:", model_path)