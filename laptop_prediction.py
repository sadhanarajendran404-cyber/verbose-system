import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("laptops.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# Custom ordinal mappings for RAM and Storage to preserve numeric scale/order
ram_map = {
    "4 GB DDR4 RAM": 4,
    "8 GB DDR4 RAM": 8,
    "8 GB Unified RAM": 8,
    "16 GB DDR4 RAM": 16,
    "16 GB DDR5 RAM": 16,
    "16 GB Unified RAM": 16,
    "32 GB DDR5 RAM": 32
}

storage_map = {
    "256 GB SSD": 256,
    "512 GB SSD": 512,
    "1 TB SSD": 1024,
    "2 TB SSD": 2048
}

data["ram"] = data["ram"].map(ram_map)
data["storage"] = data["storage"].map(storage_map)

# Label encode nominal categorical columns using separate encoders
nominal_cols = ["brand", "processor", "operating_system"]
encoders = {}

for col in nominal_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Decision Tree
# -------------------------

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

print("\nDecision Tree")
print("RMSE:", dt_rmse)
print("R2 Score:", dt_r2)

# -------------------------
# Random Forest
# -------------------------

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest")
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# -------------------------
# Support Vector Regression
# -------------------------

# SVR requires features and target to be scaled to perform optimally
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

svr = SVR(kernel="linear")
svr.fit(X_train_scaled, y_train_scaled)

svr_pred_scaled = svr.predict(X_test_scaled)
svr_pred = scaler_y.inverse_transform(svr_pred_scaled.reshape(-1, 1)).flatten()

svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
svr_r2 = r2_score(y_test, svr_pred)

print("\nSVR (with Feature & Target Scaling)")
print("RMSE:", svr_rmse)
print("R2 Score:", svr_r2)

# Best Model
scores = {
    "Decision Tree": dt_r2,
    "Random Forest": rf_r2,
    "SVR": svr_r2
}

best_model = max(scores, key=scores.get)

print("\nBest Model:", best_model)
print("Best R2 Score:", scores[best_model])