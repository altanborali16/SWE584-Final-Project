import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Get data from csv
data = pd.read_csv('./misc/heart_failure_clinical_records.csv')

# Data Information
print("Data Information:")
print(data.info())
prediction_column = 'DEATH_EVENT'

# Handling Missing Values
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Verify no missing values
print("Missing Values After Imputation:")
print(data_imputed.isnull().sum())

# Correlation Matrix
# Both logistic and random forest are same but i dont know which you will run first :)
plt.figure(figsize=(12, 10))
corr_matrix = data_imputed.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig('./misc/correlation_matrix_l.png')
plt.show()

# Train-test split
train_data, test_data = train_test_split(data_imputed, test_size=0.2, random_state=42)
# time is the most important feature, if you want to see affect you can run with it
# train_data, test_data = train_test_split(data_imputed.drop(columns=['time']), test_size=0.2, random_state=42)

# Normal Distribution of Train Data
numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()

# Both logistic and random forest are same but i dont know which you will run first :)
plt.figure(figsize=(16, 12))
for i, feature in enumerate(numeric_features):
    plt.subplot(4, 4, i+1)
    sns.histplot(train_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('./misc/feature_disturbution_l.png')
plt.show()

# Features and target
X_train = train_data.drop(columns=[prediction_column])
y_train = train_data[prediction_column]

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the logistic regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Feature Importance
feature_importance = np.abs(logreg.coef_[0])
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.savefig('./misc/feature_importance_l.png')
plt.show()

# Predictions and Evaluation
X_test = test_data.drop(columns=[prediction_column])
y_test = test_data[prediction_column]
X_test_scaled = scaler.transform(X_test)

y_pred = logreg.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))