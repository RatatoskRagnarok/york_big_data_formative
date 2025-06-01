import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score

"""
This file compares Decision Tree and Gaussian Naive Bayes classifiers
using three different train/test splits (80/20, 70/30, 60/40).

For each model and split:
- Evaluation metrics are calculated: accuracy, precision, recall, F1 score, and ROC AUC.
- Precision-recall curves are plotted for visual comparison.
- Predicted probabilities are stored and displayed in a dataframe.

A paired t-test is performed on F1 scores to assess whether the performance 
difference between models is statistically significant.

The evaluation is repeated with 15 different seeds to create 45 paired models in order to confirm t-test results.

The full code with inline dataframes and figures can be found online at:

"""
#TODO add link to jupyter notebook

# Load Data
df = pd.read_csv('hotel_final_features.csv')

# Split into independent and dependent variables

features = ['lead_time',
            'deposit_type_No Deposit', 'deposit_type_Non Refund', 'deposit_type_Refundable',
            'total_of_special_requests', 'previous_cancellations'
            ]

X = df[features]
y = df['is_canceled']

# initialise both models
tree = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42)
nb = GaussianNB()

# lists to loop through
splits = [0.2, 0.3, 0.4]
models = [tree, nb]

# lists to collect data in
accuracies = []
model_names = []
splittings = []
precisions = []
recalls = []
f1s = []
roc_aucs = []
probs = []

# set up grid for graphs
fig, axes = plt.subplots(3, 1, figsize=(8, 6))
axes = axes.flatten()

# loop each split and model
for i, split in enumerate(splits):
    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=split)
    train_pct = int((1 - split) * 100)
    # fix the 19% rounding anomaly
    if train_pct == 19:
        train_pct += 1
    test_pct = int(split * 100)

    for model in models:
        # for each model at this split
        name = model.__class__.__name__
        # fit the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # metrics
        confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = confusion.ravel()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_prob)

        # add to lists for use later
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        roc_aucs.append(roc_auc)
        probs.append(y_prob)
        model_names.append(name)
        splittings.append(f"{train_pct}/{test_pct}")

        # plot precision-recall curve
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[i].plot(rec, prec, label=f'{name} (AP = {ap:.2f})')

    axes[i].set_xlabel('Recall')
    axes[i].set_ylabel('Precision')
    axes[i].set_title(f'Precision-Recall Curve - {train_pct}/{test_pct} Split')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('Precision recall curves.png')

# Create dataframe
results_df = pd.DataFrame({
    "Model": model_names,
    "Train/Test Split": splittings,
    "Accuracy": accuracies,
    "Precision": precisions,
    "Recall (TPR)": recalls,
    "F1 Score": f1s,
    "Roc-Auc": roc_aucs
})

# do paired t-test
tree_f1s = results_df[results_df['Model']=='DecisionTreeClassifier']['F1 Score']
nb_f1s = results_df[results_df['Model']=='GaussianNB']['F1 Score']

t_stat, p_value = ttest_rel(tree_f1s, nb_f1s)

# print results
print("Results From Models")
print(results_df.round(2).sort_values(by='Model'))
print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.3f}")

# confirm result with a lot more models!
splits = [0.2, 0.3, 0.4]
seeds = range(1, 30, 2)
models = [tree, nb]

# lists to collect data in
accuracies = []
model_names = []
splittings = []
precisions = []
recalls = []
f1s = []
roc_aucs = []
probs = []
seed_nums = []

for seed in seeds:
    for model in models:
        for split in splits:
            model_names.append(model.__class__.__name__)
            seed_nums.append(seed)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=split)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])

            train_pct = int((1 - split) * 100)
            if train_pct == 19:
                train_pct += 1
            test_pct = int(split * 100)
            splittings.append(f"{train_pct}/{test_pct}")

            # confusion matrix data
            tn, fp, fn, tp = confusion.ravel()
            y_prob = model.predict_proba(X_test)[:, 1]
            probs.append(y_prob)

            # accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            # precision
            precision = precision_score(y_test, y_pred, pos_label=1)
            precisions.append(precision)

            # recall (== true positive rate)
            recall = recall_score(y_test, y_pred, pos_label=1)
            recalls.append(recall)

            # F1 score
            f1 = f1_score(y_test, y_pred, pos_label=1)
            f1s.append(f1)

            # roc-auc
            roc_auc = roc_auc_score(y_test, y_prob)
            roc_aucs.append(roc_auc)


# Create dataframe
results_df = pd.DataFrame({
    "Model": model_names,
    "Seed": seed_nums,
    "Train/Test Split": splittings,
    "Accuracy": accuracies,
    "Precision": precisions,
    "Recall (TPR)": recalls,
    "F1 Score": f1s,
    "Roc-Auc": roc_aucs
})

tree_f1s = results_df[results_df['Model']=='DecisionTreeClassifier']['F1 Score']
nb_f1s = results_df[results_df['Model']=='GaussianNB']['F1 Score']
t_stat, p_value = ttest_rel(tree_f1s, nb_f1s)

print("Results from 45 model pairs from 15 different seeds")
print(results_df.round(2).sort_values(by='Model'))
print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.3f}")
