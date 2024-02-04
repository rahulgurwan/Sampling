import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv('Creditcard_data.csv')

df.head()

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x = 'Class',data = df)
plt.show()

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
import pandas as pd

# Assuming df is your DataFrame containing the data and "Class" is the target column

# Separate features (X) and target variable (y)
features = df.drop("Class", axis=1)
target = df["Class"]

# Set random state to 47
random_state_value = 47

# Random Under-sampling
rus = RandomUnderSampler(random_state=random_state_value)
features_rus, target_rus = rus.fit_resample(features, target)
df_rus = pd.concat([features_rus, target_rus], axis=1)
df_rus.to_csv("data_rus.csv", index=False)

# Random Over-sampling
ros = RandomOverSampler(random_state=random_state_value)
features_ros, target_ros = ros.fit_resample(features, target)
df_ros = pd.concat([features_ros, target_ros], axis=1)
df_ros.to_csv("data_ros.csv", index=False)

# SMOTE Over-sampling
smote = SMOTE(random_state=random_state_value)
features_smote, target_smote = smote.fit_resample(features, target)
df_smote = pd.concat([features_smote, target_smote], axis=1)
df_smote.to_csv("data_smote.csv", index=False)

# BorderlineSMOTE Over-sampling
bs = BorderlineSMOTE(random_state=random_state_value)
features_bs, target_bs = bs.fit_resample(features, target)
df_bs = pd.concat([features_bs, target_bs], axis=1)
df_bs.to_csv("data_bs.csv", index=False)

# ADASYN Over-sampling
ad = ADASYN(random_state=random_state_value)
features_ad, target_ad = ad.fit_resample(features, target)
df_ad = pd.concat([features_ad, target_ad], axis=1)
df_ad.to_csv("data_ad.csv", index=False)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = ["data_ros.csv",
            "data_rus.csv",
            "data_smote.csv",
            "data_bs.csv",
            "data_ad.csv"]

# List of classifiers
models = [LogisticRegression(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          SVC(),
          GradientBoostingClassifier()]

# Sampling techniques
sampling_techniques = ['RandomUnderSampler', 'RandomOverSampler', 'SMOTE', 'BorderlineSMOTE', 'ADASYN']

results = []

for dataset, sampling_technique in zip(datasets, sampling_techniques):
    try:
        df = pd.read_csv(dataset)
    except FileNotFoundError:
        print(f"Error: File {dataset} not found. Check the file path.")
        continue

    X = df.drop("Class", axis=1)
    y = df["Class"]

    for model, model_name in zip(models, ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'SVC', 'GradientBoostingClassifier']):
        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model and make predictions
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy and store results
        accuracy = accuracy_score(Y_test, y_pred)
        results.append({'Sampling': sampling_technique, 'Classifier': model_name, 'Accuracy': 100 * accuracy})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
results_df.head()

pivot_df = results_df.pivot_table(index='Classifier', columns='Sampling', values='Accuracy')
pivot_df.to_csv('final.csv')

pivot_df.head()