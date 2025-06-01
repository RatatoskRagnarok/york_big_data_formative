import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

import scipy.stats as stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE

"""
This file performs various forms of feature selection

For the full jupyter notebook with inline figures and dataframes, go to

"""
#TODO add link to jupyter notebook

# Load data - two formats, one with categorical data encoded, one not
df_nocodes = pd.read_csv('hotel_cleaned_no_encoding.csv')
df_codes = pd.read_csv('hotel_cleaned.csv')

# make lists of columns
num_cols = []
cat_cols = []

for col in df_nocodes.columns:  # arbitrary cut off, some of the num_cols are still sort of categorical
    if (df_nocodes[col].dtype=='object') or (df_nocodes[col].nunique() <= 12):  # to catch arrival month
        cat_cols.append(col)
    else:
        num_cols.append(col)

cat_cols.remove('is_canceled')  # remove target variable

# because it's easier to see that it's noise
num_cols.append('children')

# visual analysis

# bar charts for categoricals

# Set up grid layout
cols_per_row = 3
num_plots = len(cat_cols)
rows = math.ceil(num_plots / cols_per_row)

fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    grouped = df_nocodes.groupby(col)['is_canceled'].mean().sort_values(ascending=False)
    sns.barplot(x=grouped.index, y=grouped.values, ax=axes[i])
    axes[i].set_ylabel('Number of Cancellations')
    axes[i].set_title(f'Cancellations by {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# stripplots for numericals

# Set up grid layout
cols_per_row = 3
num_plots = len(num_cols)
rows = math.ceil(num_plots / cols_per_row)

fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.stripplot(x='is_canceled', y=col, data=df_nocodes, alpha=0.3, jitter=True, ax=axes[i])
    axes[i].set_title(f'{col} vs is_canceled')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# Notable plots

notable_cat = ['deposit_type', 'total_of_special_requests', 'is_repeated_guest']
notable_nums = ['lead_time', 'previous_cancellations', 'previous_bookings_not_canceled']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(notable_cat):
    grouped = df_nocodes.groupby(col)['is_canceled'].mean().sort_values(ascending=False)
    sns.barplot(x=grouped.index, y=grouped.values, ax=axes[i])
    axes[i].set_ylabel('Number of Cancellations')
    axes[i].set_title(f'Cancellations by {col}')
    axes[i].tick_params(axis='x', rotation=45)

for i, col in enumerate(notable_nums):
    sns.stripplot(x='is_canceled', y=col, data=df_nocodes, alpha=0.3, jitter=True, ax=axes[i+3])
    axes[i+3].set_title(f'{col} vs is_canceled')

plt.tight_layout()
plt.savefig("visual_analysis.png")

# correlations
# Pearson for numerical features
p_values = []
c_values = {}

for col in num_cols:
    corr, p_val = stats.pearsonr(df_nocodes[col], df_nocodes['is_canceled'])
    p_values.append(p_val)
    c_values[col] = corr

p_values_df = pd.DataFrame.from_dict(c_values, orient='index', columns=['Pearson Correlation'])
p_values_df['p-value'] = p_values
p_values_df['Significant'] = p_values_df['p-value'].apply(lambda x: 'Yes' if x < 0.05 else 'No')

print("\nTop 10 Pearson Correlations for Numerical Features")
print(p_values_df.sort_values(by=['Pearson Correlation'], ascending=False).head(10))

# chi-square for categorical features
chi_square_results = {}

for col in cat_cols:
    contingency_table = pd.crosstab(df_nocodes[col], df_nocodes['is_canceled'])
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)
    chi_square_results[col] = {'chi2_stat': chi2_stat,
                               'p_value': p_val,
                               'degrees_of_freedom': dof,
                               'sample_size': contingency_table.values.sum()
                               }


chi_square_df = pd.DataFrame.from_dict(chi_square_results, orient='index')
chi_square_df['Significant'] = chi_square_df['p_value'].apply(lambda x: 'Yes' if x < 0.05 else 'No')
chi_square_df.rename(columns={'chi2_stat': 'Chi-Square Statistic', 'p_value': 'p-value', 'degrees_of_freedom': 'DOF', 'sample_size': 'N'}, inplace=True)


print("\nTop 10 Chi-Square results for Categorical Features")
print(chi_square_df.sort_values(by=['Chi-Square Statistic'], ascending=False).head(10))

# Random Forest for feature importance
X = df_codes.drop(columns=['is_canceled'])
y = df_codes['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Cross-validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)
all_importances = []

rankings = []
accuracies = []
precisions = []
f1s = []
recalls = []

# Evaluate the model and get feature importances across cross-validation splits
for train_index, val_index in kf.split(X_train):
    X_kf_train, X_kf_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_kf_train, y_kf_val = y_train.iloc[train_index], y_train.iloc[val_index]

    rf_model.fit(X_kf_train, y_kf_train)
    importances = rf_model.feature_importances_
    all_importances.append(importances)

    y_pred = rf_model.predict(X_kf_val)

    accuracies.append(accuracy_score(y_kf_val, y_pred))
    precisions.append(precision_score(y_kf_val, y_pred, zero_division=np.nan))
    f1s.append(f1_score(y_kf_val, y_pred, zero_division=np.nan))
    recalls.append(recall_score(y_kf_val, y_pred, zero_division=np.nan))
    conf = confusion_matrix(y_kf_val, y_pred)


all_importances_df = pd.DataFrame(all_importances, columns=X.columns)

summary_importance_df = all_importances_df.T

summary_importance_df['Mean Importance'] = all_importances_df.mean()
summary_importance_df['Std Dev Importance'] = all_importances_df.std()

summary_importance_df = summary_importance_df.sort_values(by='Mean Importance', ascending=False)
feature_importances = summary_importance_df['Mean Importance']

top_features = summary_importance_df.head(15)

print("\nTop 15 features ranked by Random Forest Feature Importance")
print(top_features)

# graph of top features
plt.figure(figsize=(10, 5))
plt.barh(top_features.index, top_features["Mean Importance"])
plt.gca().invert_yaxis()  # Highest importance at the top
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features Ranked by Random Forest Feature Importance")

plt.savefig("RandomForest Features.png")

# scores from Random Forest classifiers
scores = np.array([accuracies, precisions, recalls, f1s]).T
df_scores = pd.DataFrame(scores, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
df_scores.loc['Mean'] = df_scores.mean()

print("\nEvaluation metrics of Random Forest models used for feature importance")
print(df_scores)

# RFE with random forest
# Could not be run on all features for performance reasons
# Selection of features explained in text
features = ['lead_time', 'previous_cancellations', 'adults', 'days_in_waiting_list', 'adr',
            'deposit_type_No Deposit', 'deposit_type_Non Refund', 'deposit_type_Refundable',
            'total_of_special_requests', 'market_segment_Aviation', 'market_segment_Complementary',
            'market_segment_Corporate', 'market_segment_Direct', 'market_segment_Groups',
            'market_segment_Offline TA/TO', 'market_segment_Online TA', 'required_car_parking_spaces',
            'arrival_month'
            ]

X = df_codes[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)
all_importances = []

# Initialize RFE with Random Forest
rfe = RFE(estimator=rf_model, n_features_to_select=5)

rankings = []

# Evaluate the model and get feature importances across cross-validation splits
for train_index, val_index in kf.split(X_train):
    X_kf_train, X_kf_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_kf_train, y_kf_val = y_train.iloc[train_index], y_train.iloc[val_index]

    rfe.fit(X_kf_train, y_kf_train)
    rfe_rank = rfe.ranking_
    rankings.append(rfe_rank)

all_rankings_df = pd.DataFrame(rankings, columns=X.columns)

summary_importance_df = all_rankings_df.T

summary_importance_df['Mean Ranking'] = all_rankings_df.mean()
summary_importance_df['Std Dev Ranking'] = all_rankings_df.std()

summary_importance_df = summary_importance_df.sort_values(by='Mean Ranking')

print("\nTop 10 Rankings from RFE with Random Forest")
print(summary_importance_df.head(10))

# final feature choice
# check for redundancy
features = ['lead_time',
            'deposit_type_No Deposit', 'deposit_type_Non Refund', 'deposit_type_Refundable',
            'total_of_special_requests', 'market_segment_Aviation', 'market_segment_Complementary',
            'market_segment_Corporate', 'market_segment_Direct', 'market_segment_Groups',
            'market_segment_Offline TA/TO', 'market_segment_Online TA', 'is_canceled'
            ]

X = df_codes[features]

plt.figure(figsize=(12, 8))

correlation_matrix_pearson = df_codes[features].corr(method='pearson')
sns.heatmap(correlation_matrix_pearson, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Heatmap')

# Adjust layout
plt.tight_layout()
plt.savefig("Redundancy Found.png")

# replace feature, check again
features = ['lead_time',
            'deposit_type_No Deposit', 'deposit_type_Non Refund', 'deposit_type_Refundable',
            'total_of_special_requests', 'previous_cancellations', 'is_canceled'
            ]

X = df_codes[features]

plt.figure(figsize=(12, 8))

correlation_matrix_pearson = df_codes[features].corr(method='pearson')
sns.heatmap(correlation_matrix_pearson, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Heatmap')

# Adjust layout
plt.tight_layout()
plt.savefig("final features correlation.png")

# save copy of data with final chosen features
features = ['lead_time',
            'deposit_type_No Deposit', 'deposit_type_Non Refund', 'deposit_type_Refundable',
            'total_of_special_requests', 'previous_cancellations', 'is_canceled'
            ]

df = df_codes[features]

# save final version
df.to_csv("hotel_final_features.csv", index=False)
