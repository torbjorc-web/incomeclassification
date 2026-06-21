import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
df = pd.read_csv('adult.data', header=None, names=col_names)

# Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()
print(df.head())

# ============================================
# TASK 1: Check Class Imbalance
# ============================================
print("=== TASK 1: Check Class Imbalance ===")
income_distribution = df['income'].value_counts()
print("Income distribution:")
print(income_distribution)
print(f"\nPercentage of >50K: {df['income'].value_counts(normalize=True)['>50K'] * 100:.2f}%")
print(f"Percentage of <=50K: {df['income'].value_counts(normalize=True)['<=50K'] * 100:.2f}%")

# Check if imbalanced
is_imbalanced = df['income'].value_counts(normalize=True).min() < 0.4
print(f"\nIs dataset imbalanced? {is_imbalanced}")


# ============================================
# TASK 2: Create feature dataframe X with dummy variables
# ============================================
print("\n=== TASK 2: Create Feature DataFrame X ===")
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education']

# Create dummy variables for categorical features
X = pd.get_dummies(df[feature_cols], drop_first=False)
print(f"Number of features after dummy encoding: {X.shape[1]}")
print(f"Features: {X.columns.tolist()}")


# ============================================
# TASK 3: Create heatmap of X data to see feature correlation
# ============================================
print("\n=== TASK 3: Correlation Heatmap ===")
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Predictor Variables')
plt.tight_layout()
plt.show()


# ============================================
# TASK 4: Create output variable y (binary)
# ============================================
print("\n=== TASK 4: Create Output Variable y ===")
y = df['income'].map({'<=50K': 0, '>50K': 1})
print(f"y variable created successfully")
print(f"y distribution: {y.value_counts()}")
print(f"Percentage of 1 ('>50K'): {y.mean() * 100:.2f}%")


# ============================================
# TASK 5a: Split data into train and test set
# ============================================
print("\n=== TASK 5a: Train-Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


# ============================================
# TASK 5b: Fit LR model and predict on test set
# ============================================
print("\n=== TASK 5b: Fit Logistic Regression Model ===")

# Scale the data (recommended for L1 penalty)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression model with specified parameters
log_reg = LogisticRegression(C=0.05, penalty='l1', solver='liblinear', max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Create predictions
y_pred = log_reg.predict(X_test_scaled)
print(f"Model fitted and predictions created successfully")


# ============================================
# TASK 6: Print model parameters (intercept and coefficients)
# ============================================
print("\n=== TASK 6: Model Parameters ===")
print('Model Parameters, Intercept:')
print(log_reg.intercept_[0])

print('Model Parameters, Coeff:')
print(log_reg.coef_[0])


# ============================================
# TASK 7: Evaluate predictions - confusion matrix and accuracy
# ============================================
print("\n=== TASK 7: Model Evaluation ===")
print('Confusion Matrix on test set:')
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print('Accuracy Score on test set:')
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)


# ============================================
# TASK 8: Create DataFrame of coefficients and variable names, sorted
# ============================================
print("\n=== TASK 8: Coefficient DataFrame ===")
coef_df = pd.DataFrame({
    'variable': X.columns,
    'coefficient': log_reg.coef_[0]
})

# Sort by coefficient and exclude zeros
coef_df_sorted = coef_df[coef_df['coefficient'] != 0].sort_values('coefficient')
print(f"Number of non-zero coefficients: {len(coef_df_sorted)}")
print("\nNon-zero coefficients (sorted):")
print(coef_df_sorted)


# ============================================
# TASK 9: Barplot of coefficients sorted in ascending order
# ============================================
print("\n=== TASK 9: Coefficient Barplot ===")
plt.figure(figsize=(12, 8))
plt.barh(coef_df_sorted['variable'], coef_df_sorted['coefficient'], 
         color='steelblue', edgecolor='black', alpha=0.8)
plt.xlabel('Coefficient Value')
plt.ylabel('Variable')
plt.title('Logistic Regression Coefficients (Sorted Ascending)')
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
plt.tight_layout()
plt.show()


# ============================================
# TASK 10: Plot ROC curve and print AUC value
# ============================================
print("\n=== TASK 10: ROC Curve and AUC ===")
# Get predicted probabilities
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Calculate AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Value: {roc_auc:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Income Classification')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("=== ALL 10 TASKS COMPLETE ===")
print("="*60)
print(f"✓ Dataset imbalanced: {is_imbalanced}")
print(f"✓ Features after dummy encoding: {X.shape[1]}")
print(f"✓ Training size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}")
print(f"✓ Model accuracy: {acc_score:.4f} ({acc_score * 100:.2f}%)")
print(f"✓ Non-zero coefficients: {len(coef_df_sorted)}")
print(f"✓ AUC score: {roc_auc:.4f} ({roc_auc * 100:.2f}%)")
