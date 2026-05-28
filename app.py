import pandas as pd, numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
app = Flask(__name__)
df = pd.read_csv("laptops.csv")
# Coerce price to numeric, converting headers or git conflict markers to NaN
df["price"] = pd.to_numeric(df["price"], errors="coerce")
# Dynamic mapping of RAM and storage strings to numerical values with robust error handling
RAM_MAP = {}
for x in df["ram"].dropna().unique():
    try:
        RAM_MAP[x] = int(str(x).split()[0])
    except Exception:
        pass
STORAGE_MAP = {}
for x in df["storage"].dropna().unique():
    try:
        value = int(str(x).split()[0])
        if "TB" in str(x):
            value *= 1024
        STORAGE_MAP[x] = value
    except Exception:
        pass
m_df = df.copy()
m_df["ram"], m_df["storage"] = m_df["ram"].map(RAM_MAP), m_df["storage"].map(STORAGE_MAP)
# Drop any invalid rows, duplicate headers, or git conflict markers from the training set
m_df = m_df.dropna()
encoders = {c: LabelEncoder().fit(m_df[c].astype(str)) for c in ["brand", "processor", "operating_system"]}
for c, le in encoders.items(): m_df[c] = le.transform(m_df[c].astype(str))
X, y = m_df.drop(columns="price"), m_df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeRegressor(random_state=42).fit(X_tr, y_tr)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
sc_X, sc_y = StandardScaler(), StandardScaler()
X_tr_s = sc_X.fit_transform(X_tr)
y_tr_s = sc_y.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
svr = SVR(kernel="linear").fit(X_tr_s, y_tr_s)
def get_m(m, is_svr=False):
    p = m.predict(sc_X.transform(X_te) if is_svr else X_te)
    if is_svr: 
      p = sc_y.inverse_transform(p.reshape(-1, 1)).flatten()
    return {
         "rmse": float(np.sqrt(mean_squared_error(y_te, p))), "r2": float(r2_score(y_te, p))
         }
metrics = {"Decision Tree": get_m(dt), "Random Forest": get_m(rf), "SVR": get_m(svr, 1)}
@app.route("/")
def index(): return render_template("index.html")
@app.route("/api/options")
def get_options():
    options = {}
    for c in X.columns:
        if c in encoders:
            options[c] = sorted(encoders[c].classes_.tolist())
        elif c == 'ram':
            options[c] = sorted(RAM_MAP.keys(), key=RAM_MAP.get)
        elif c == 'storage':
            options[c] = sorted(STORAGE_MAP.keys(), key=STORAGE_MAP.get)
    return jsonify(options)
@app.route("/api/metrics")
def get_metrics_endpoint(): return jsonify(metrics)
@app.route("/api/stats")
def get_stats():
    # Use clean dataframe indices to build clean statistics, avoiding git conflict marker lines
    df_c = df.loc[m_df.index]
    get_avg = lambda col, sort=None: (lambda g: {"labels": g.index.tolist(), "values": [round(v, 2) for v in g.tolist()]})(df_c.groupby(col)["price"].mean().reindex(sorted(df_c[col].unique().tolist(), key=sort)) if sort else df_c.groupby(col)["price"].mean())
    p_cnt = df_c["processor"].value_counts()
    return jsonify({"brand_prices": get_avg("brand"), "ram_prices": get_avg("ram", RAM_MAP.get), "storage_prices": get_avg("storage", STORAGE_MAP.get), "processor_counts": {"labels": p_cnt.index.tolist(), "values": p_cnt.tolist()}})
@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.json
    feat = [[
         encoders[c].transform([d[c]])[0] 
         if c in encoders 
         else (RAM_MAP[d[c]] if c=='ram' else STORAGE_MAP[d[c]]) 
         for c in X.columns
    ]]
    svr_p = sc_y.inverse_transform(
         svr.predict(sc_X.transform(feat)).reshape(-1, 1)
         )[0][0]
    return jsonify({
        "predicted_price": round(float(svr_p), 2), 
        "svr_price": round(float(svr_p), 2),
        "dt_price": round(float(dt.predict(feat)[0]), 2), 
        "rf_price": round(float(rf.predict(feat)[0]), 2)
    })
import os
if __name__ == "__main__":
    app.run(host="0.0.0.0", 
port=int(os.environ.get("PORT", 5000)))
