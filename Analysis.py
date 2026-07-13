# Part 2 - Supervised ML Model: Build, Train, Evaluate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)


# ===== STEP 0: CREATE DUMMY DATA BECAUSE I DON'T HAVE raw_data.csv =====
data = {
    'Category': ['A','B','A','B','C','A','B','C','A','B'],
    'Sales': [100, 150, None, 200, 250, 120, 180, 220, None, 160],
    'Profit': [20, 30, 25, None, 40, 22, 35, 38, 28, None],
    'Date': [1,2,3,4,5,6,7,8,9,10]
}
df = pd.DataFrame(data)
df.to_csv("raw_data.csv", index=False)
print("Dummy raw_data.csv created!")

# ===== FROM HERE YOUR ORIGINAL CODE STARTS =====
# 1. LOAD DATASET
df = pd.read_csv("raw_data.csv") # this will now work
print("First 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nShape:", df.shape)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import *

# 1. LOAD DATA
df = pd.read_csv("cleaned_data.csv")
print("Data shape:", df.shape)
print(df.head())

# CHANGE THIS to your target column name
TARGET_REG = 'Sales' # <-- replace 'Sales' with your column

# Drop rows with NaN in target
df = df.dropna(subset=[TARGET_REG])

# Separate features and target
X = df.drop(TARGET_REG, axis=1)
y_reg = df[TARGET_REG]

# Handle categorical columns if any
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Scale features for Ridge/Lasso/Logistic
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*50)
print("1. REGRESSION MODELS")
print("="*50)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
pred_ridge = ridge.predict(X_test_scaled)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
pred_lasso = lasso.predict(X_test_scaled)

# Results table
results = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Lasso'],
    'MSE': [mean_squared_error(y_test, pred_lr), mean_squared_error(y_test, pred_ridge), mean_squared_error(y_test, pred_lasso)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, pred_lr)), np.sqrt(mean_squared_error(y_test, pred_ridge)), np.sqrt(mean_squared_error(y_test, pred_lasso))],
    'R2': [r2_score(y_test, pred_lr), r2_score(y_test, pred_ridge), r2_score(y_test, pred_lasso)]
})
print(results)

# Top 3 features by coefficient
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
print("\nTop 3 Features by |Coefficient|:")
print(coef_df.sort_values('Abs_Coef', ascending=False).head(3))
print("\nBottom 3 Features by |Coefficient|:")
print(coef_df.sort_values('Abs_Coef', ascending=True).head(3))


print("\n" + "="*50)
print("2. CLASSIFICATION MODEL - LOGISTIC REGRESSION")
print("="*50)

# Create binary target: 1 if above median, 0 otherwise
median_val = y_reg.median()
y_class = (y_reg > median_val).astype(int)
print(f"Class distribution:\n{y_class.value_counts()}")

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

Xc_train_scaled = scaler.fit_transform(Xc_train)
Xc_test_scaled = scaler.transform(Xc_test)

# Baseline Logistic Regression C=1.0
logreg_1 = LogisticRegression(C=1.0, max_iter=1000)
logreg_1.fit(Xc_train_scaled, yc_train)
proba_1 = logreg_1.predict_proba(Xc_test_scaled)[:,1]
pred_1 = logreg_1.predict(Xc_test_scaled)

print("\nBaseline C=1.0:")
print("Accuracy:", accuracy_score(yc_test, pred_1))
print("Precision:", precision_score(yc_test, pred_1))
print("Recall:", recall_score(yc_test, pred_1))
print("F1:", f1_score(yc_test, pred_1))
print("ROC AUC:", roc_auc_score(yc_test, proba_1))
print("\nConfusion Matrix:\n", confusion_matrix(yc_test, pred_1))

# ROC Curve
fpr, tpr, _ = roc_curve(yc_test, proba_1)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(yc_test, proba_1):.3f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic C=1.0')
plt.legend()
plt.savefig('roc_curve.png')
plt.show()

# Threshold experiment
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
thresh_results = []
for t in thresholds:
    pred_t = (proba_1 >= t).astype(int)
    thresh_results.append({
        'Threshold': t,
        'Precision': precision_score(yc_test, pred_t),
        'Recall': recall_score(yc_test, pred_t),
        'F1': f1_score(yc_test, pred_t)
    })
thresh_df = pd.DataFrame(thresh_results)
print("\nThreshold Tuning Results:")
print(thresh_df)

best_row = thresh_df.loc[thresh_df['F1'].idxmax()]
print(f"\nBest Threshold: {best_row['Threshold']} with F1 = {best_row['F1']:.3f}")

# Regularization experiment C=0.01
logreg_001 = LogisticRegression(C=0.01, max_iter=1000)
logreg_001.fit(Xc_train_scaled, yc_train)
proba_001 = logreg_001.predict_proba(Xc_test_scaled)[:,1]
pred_001 = logreg_001.predict(Xc_test_scaled)

comp_df = pd.DataFrame({
    'Model': ['C=1.0', 'C=0.01'],
    'Precision': [precision_score(yc_test, pred_1), precision_score(yc_test, pred_001)],
    'Recall': [recall_score(yc_test, pred_1), recall_score(yc_test, pred_001)],
    'F1': [f1_score(yc_test, pred_1), f1_score(yc_test, pred_001)],
    'AUC': [roc_auc_score(yc_test, proba_1), roc_auc_score(yc_test, proba_001)]
})
print("\nRegularization Comparison:")
print(comp_df)


print("\n" + "="*50)
print("3. BOOTSTRAP CI FOR AUC DIFFERENCE")
print("="*50)

n_bootstrap = 500
auc_diffs = []
rng = np.random.RandomState(42)

for i in range(n_bootstrap):
    indices = rng.choice(len(yc_test), len(yc_test), replace=True)
    y_true_bs = yc_test.iloc[indices]
    proba1_bs = proba_1[indices]
    proba001_bs = proba_001[indices]
    auc1 = roc_auc_score(y_true_bs, proba1_bs)
    auc001 = roc_auc_score(y_true_bs, proba001_bs)
    auc_diffs.append(auc1 - auc001)

mean_diff = np.mean(auc_diffs)
ci_lower = np.percentile(auc_diffs, 2.5)
ci_upper = np.percentile(auc_diffs, 97.5)

print(f"Mean AUC difference: {mean_diff:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
if ci_lower > 0:
    print("Interpretation: C=1.0 is significantly better than C=0.01")
else:
    print("Interpretation: Difference is not statistically significant")

print("\nDone! Save results to README.md")