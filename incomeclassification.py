import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('adult.data', header=None, names=col_names)

# Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()
print(df.head())

# 1. Check Class Imbalance
print('Class imbalance:')
print(df['income'].value_counts())
print(df['income'].value_counts(normalize=True))

# 2. Create feature dataframe X with feature columns and dummy variables for categorical features
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education']
X = pd.get_dummies(df[feature_cols], drop_first=False)

# 3. Create a heatmap of X data to see feature correlation
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# 4. Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greater than 50k
y = df['income'].map({'<=50K': 0, '>50K': 1})

# 5a. Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 5b. Fit LR model with sklearn on train set, and predicting on the test set
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

log_reg = LogisticRegress
