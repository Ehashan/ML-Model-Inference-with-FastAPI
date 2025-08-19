import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("Iris.csv")

# Features and target
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# Encode target labels
target_names = y.unique().tolist()
y_encoded = y.replace({name: idx for idx, name in enumerate(target_names)})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# Save model, scaler, and metadata
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump({"feature_names": X.columns.tolist(), "target_names": target_names}, "model_metadata.pkl")

print("✅ Model, scaler, and metadata saved successfully!")
