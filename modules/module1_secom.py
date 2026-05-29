import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
# SECOM: 1567 wafers, 591 sensor readings each
# Labels: -1 = pass, 1 = fail
X = pd.read_csv('data/secom.data', sep=' ', header=None)
y_raw = pd.read_csv('data/secom_labels.data', sep=' ', header=None)
y = y_raw[0].map({-1: 0, 1: 1})  # convert to 0=pass, 1=fail

print(f"Wafers: {X.shape[0]}, Sensors: {X.shape[1]}")
print(f"Pass: {(y==0).sum()}, Fail: {(y==1).sum()}")

# ── 2. CLEAN DATA ─────────────────────────────────────────────────────────────
missing_pct = X.isnull().mean()
X = X.loc[:, missing_pct < 0.4]
X = X.fillna(X.median())
X = X.loc[:, X.std() > 0]

# ── 3. SCALE + BALANCE ────────────────────────────────────────────────────────
# StandardScaler: makes all sensors comparable (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE: creates synthetic fail samples so model sees balanced classes
# Without this, model just predicts "pass" for everything
print("Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_balanced, y_balanced = sm.fit_resample(X_scaled, y)
print(f"After SMOTE - Pass: {(y_balanced==0).sum()}, Fail: {(y_balanced==1).sum()}")

# ── 4. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# ── 5. TRAIN MODEL ────────────────────────────────────────────────────────────
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 6. EVALUATE ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=['Pass', 'Fail']))

# ── 7. CONFUSION MATRIX PLOT ─────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pass', 'Fail']); ax.set_yticklabels(['Pass', 'Fail'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('SECOM Fault Detection — Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', 
                color='white' if cm[i, j] > cm.max()/2 else 'black',
                fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module1_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/module1_confusion_matrix.png")

# ── 8. SHAP FEATURE IMPORTANCE ────────────────────────────────────────────────
# SHAP tells us WHICH sensors actually drive failures
# This is what a yield engineer needs - not just "model says fail"
print("\nCalculating SHAP values (this takes ~30 seconds)...")
explainer = shap.TreeExplainer(model)
# Use a sample of 200 to keep it fast
sample_idx = np.random.choice(len(X_test), size=200, replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])

# Top 15 most important sensors
feature_names = [f'Sensor_{i}' for i in X.columns]
shap_vals_fail = shap_values[1] if isinstance(shap_values, list) else shap_values
mean_shap = np.abs(shap_vals_fail).mean(axis=0)
top15_idx = np.argsort(mean_shap)[-15:]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(range(15), mean_shap[top15_idx], color='steelblue')
ax.set_yticks(range(15))
ax.set_yticklabels([feature_names[i] for i in top15_idx], fontsize=9)
ax.set_xlabel('Mean |SHAP Value| — Impact on Failure Prediction')
ax.set_title('Top 15 Sensors Driving Wafer Failures\n(Higher = More Impact on Yield)')
plt.tight_layout()
plt.savefig('outputs/module1_shap_importance.png', dpi=150)
plt.close()
print("Saved: outputs/module1_shap_importance.png")

print("\nModule 1 complete.")