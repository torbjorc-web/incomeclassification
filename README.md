Income Classification using Logistic Regression
This is the line-by-line Codecademy-style solution for the Census Income project.

What it does
Loads the Adult Census dataset from adult.data.

Cleans whitespace from text columns.

Checks class imbalance.

Builds X with dummy variables.

Plots a correlation heatmap.

Creates binary target y.

Splits into train and test sets.

Scales features before fitting L1 logistic regression.

Evaluates the model with confusion matrix, accuracy, and ROC-AUC.

Plots coefficient and ROC graphs.

Notes
codecademylib3 is included because that is often required in Codecademy environments.

Scaling is used because L1 regularization is sensitive to feature scale.
