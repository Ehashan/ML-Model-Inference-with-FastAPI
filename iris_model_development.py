from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model + metadata
joblib.dump(model, "model.pkl")
joblib.dump({
    "feature_names": iris.feature_names,
    "target_names": iris.target_names
}, "model_metadata.pkl")

print("✅ Model and metadata saved successfully!")