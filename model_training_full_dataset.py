import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = "./data/preprocessed_dataset.csv"
df = pd.read_csv(file_path)

# Encode categorical variable
if df['Sex'].dtype == object:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Define target and features
data_y = df['target']
data_x = df.drop('target', axis=1)

# Feature selection using RandomForest
Ran_state = 50
rf_selector = RandomForestClassifier(n_estimators=200, random_state=Ran_state, n_jobs=-1)
rf_selector.fit(data_x, data_y)
selector = SelectFromModel(rf_selector, prefit=True)
data_x = pd.DataFrame(selector.transform(data_x), columns=data_x.columns[selector.get_support()])
selected_features = data_x.columns

# Define models and hyperparameter grids
SVM = SVC(class_weight='balanced', probability=True, random_state=Ran_state)
RF = RandomForestClassifier(class_weight='balanced', random_state=Ran_state, n_jobs=-1)
XGB = XGBClassifier(objective='multi:softprob', random_state=Ran_state, n_jobs=-1)

SVM_grid = {
    'C': [0.1, 10, 80],
    'gamma': ['scale', 0.1, 1, 50],
    'kernel': ['rbf'],
}

RF_grid = {
    'n_estimators': [50, 100, 200],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.7],
    'bootstrap': [True],
    'max_depth': [None, 5, 10, 20, 30],
    'class_weight': ['balanced'],
}

XGB_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1, 0.3, 0.8],
    'max_depth': [7, 8],
    'subsample': [0.3, 0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
}

gridlist = [SVM_grid, RF_grid, XGB_grid]
classlist = [SVM, RF, XGB]
k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=Ran_state)

# Grid search function
def gridsearch(gridlist, classlist, data_x, data_y, cv):
    best_estimators = []
    for i in range(len(gridlist)):
        grid = GridSearchCV(classlist[i], gridlist[i], scoring='roc_auc_ovr', cv=cv, n_jobs=-1, error_score='raise')
        grid.fit(data_x, data_y)
        best_estimators.append(grid.best_estimator_)
    return best_estimators

# Execute grid search
best_classlist = gridsearch(gridlist, classlist, data_x, data_y, cv)

# Evaluate models
metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
std_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
fold_metrics = []

for model, name in zip(best_classlist, ['SVM', 'RF', 'XGB']):
    fold_results = {'Fold': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
    for fold, (train_idx, val_idx) in enumerate(cv.split(data_x, data_y), 1):
        model.fit(data_x.iloc[train_idx], data_y.iloc[train_idx])
        pred = model.predict(data_x.iloc[val_idx])
        y_score = model.predict_proba(data_x.iloc[val_idx])
        acc = accuracy_score(data_y.iloc[val_idx], pred)
        prec = precision_score(data_y.iloc[val_idx], pred, average='weighted')
        rec = recall_score(data_y.iloc[val_idx], pred, average='weighted')
        f1 = f1_score(data_y.iloc[val_idx], pred, average='weighted')
        auc_score = roc_auc_score(label_binarize(data_y.iloc[val_idx], classes=np.unique(data_y)), y_score, multi_class='ovr')
        cm = confusion_matrix(data_y.iloc[val_idx], pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        avg_TPR = np.mean(TPR)
        avg_TNR = np.mean(TNR)
        fold_results['Fold'].append(f'Fold {fold}')
        fold_results['Accuracy'].append(acc)
        fold_results['Precision'].append(prec)
        fold_results['Recall'].append(rec)
        fold_results['F1-score'].append(f1)
        fold_results['AUC'].append(auc_score)
        fold_results['TPR'].append(avg_TPR)
        fold_results['TNR'].append(avg_TNR)
    fold_df = pd.DataFrame(fold_results)
    fold_df['Model'] = name
    fold_metrics.append(fold_df)

all_fold_metrics = pd.concat(fold_metrics)

for model_name, group in all_fold_metrics.groupby('Model'):
    metrics['Model'].append(model_name)
    metrics['Accuracy'].append(group['Accuracy'].mean())
    metrics['Precision'].append(group['Precision'].mean())
    metrics['Recall'].append(group['Recall'].mean())
    metrics['F1-score'].append(group['F1-score'].mean())
    metrics['AUC'].append(group['AUC'].mean())
    metrics['TPR'].append(group['TPR'].mean())
    metrics['TNR'].append(group['TNR'].mean())
    std_metrics['Accuracy'].append(group['Accuracy'].std())
    std_metrics['Precision'].append(group['Precision'].std())
    std_metrics['Recall'].append(group['Recall'].std())
    std_metrics['F1-score'].append(group['F1-score'].std())
    std_metrics['AUC'].append(group['AUC'].std())
    std_metrics['TPR'].append(group['TPR'].std())
    std_metrics['TNR'].append(group['TNR'].std())

performance_metrics = pd.DataFrame(metrics)
std_performance_metrics = pd.DataFrame(std_metrics, index=performance_metrics['Model'])

# Save metrics
os.makedirs("./results", exist_ok=True)
performance_metrics.to_csv("./results/overall_model_performance.csv", index=False)
std_performance_metrics.to_csv("./results/overall_model_performance_std.csv")

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(performance_metrics.set_index('Model'), annot=True, fmt=".4f", cmap='Greys')
plt.title('Performance Metrics across Models (Mean)', fontsize=16, fontname='Times New Roman')
plt.tight_layout()
plt.savefig("./results/heatmap_performance_mean.png")

plt.figure(figsize=(10, 6))
sns.heatmap(std_performance_metrics, annot=True, fmt=".4f", cmap='Greys')
plt.title('Performance Metrics across Models (Standard Deviation)', fontsize=16, fontname='Times New Roman')
plt.tight_layout()
plt.savefig("./results/heatmap_performance_std.png")

# Feature importance
cv_importance = StratifiedKFold(n_splits=5, shuffle=True, random_state=Ran_state)
fold_feature_importances = {'RF': [], 'XGB': [], 'SVM': []}
for model, name in zip(best_classlist, ['SVM', 'RF', 'XGB']):
    for train_idx, val_idx in cv_importance.split(data_x, data_y):
        model.fit(data_x.iloc[train_idx], data_y.iloc[train_idx])
        if name in ['RF', 'XGB']:
            importances = model.feature_importances_
        else:
            result = permutation_importance(model, data_x.iloc[val_idx], data_y.iloc[val_idx], 
                                            n_repeats=5, random_state=Ran_state, n_jobs=-1)
            importances = result.importances_mean
        fold_feature_importances[name].append(importances)

mean_feature_importances = {name: np.mean(importances, axis=0)
                            for name, importances in fold_feature_importances.items()}
feature_importances_df = pd.DataFrame(mean_feature_importances, index=selected_features)
feature_importances_df.to_csv("./results/feature_importance_matrix.csv")

# Plot average feature importance
mean_importances = feature_importances_df.mean(axis=1).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_importances.values, y=mean_importances.index, palette="Greys_r")
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.title("Average Feature Importance", fontsize=16)
plt.tight_layout()
plt.savefig("./results/feature_importance_avg.png")

# Plot confusion matrices
fig, axs = plt.subplots(1, 3, figsize=(15, 7))
for ax, model, name in zip(axs.flatten(), best_classlist, ['SVM', 'RF', 'XGB']):
    all_preds, all_actuals = [], []
    for train_idx, test_idx in cv.split(data_x, data_y):
        model.fit(data_x.iloc[train_idx], data_y.iloc[train_idx])
        preds = model.predict(data_x.iloc[test_idx])
        all_preds.extend(preds)
        all_actuals.extend(data_y.iloc[test_idx])
    final_cm = confusion_matrix(all_actuals, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=final_cm)
    disp.plot(ax=ax, cmap='Greys', colorbar=False)
    ax.set_title(f'{name}', fontsize=16)
plt.tight_layout()
plt.savefig("./results/confusion_matrices.png")
