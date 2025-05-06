import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
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

# Load dataset
file_path = "./data/your_dataset.csv"
df = pd.read_csv(file_path)

# Encode categorical variable
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Define target and features
data_y = df['target']
data_x = df.drop('target', axis=1)

# Remove outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=50, n_jobs=-1)
outliers_pred = iso_forest.fit_predict(data_x)
data_x_cleaned = data_x.iloc[outliers_pred == 1]
data_y_cleaned = data_y.iloc[outliers_pred == 1]

# Feature scaling
scaler = MinMaxScaler()
data_x_scaled = scaler.fit_transform(data_x_cleaned)
data_x_cleaned = pd.DataFrame(data_x_scaled, columns=data_x_cleaned.columns)

# Define models
Ran_state = 50
RF = RandomForestClassifier(class_weight='balanced', random_state=Ran_state, n_jobs=-1)
XGB = XGBClassifier(objective='multi:softprob', random_state=Ran_state, n_jobs=-1)
SVM = SVC(class_weight='balanced', probability=True, random_state=Ran_state)

# Hyperparameter grids
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

# Feature selection using RandomForest
rf_selector = RandomForestClassifier(n_estimators=200, random_state=Ran_state, n_jobs=-1)
rf_selector.fit(data_x_cleaned, data_y_cleaned)
selector = SelectFromModel(rf_selector, prefit=True)
data_x_cleaned = pd.DataFrame(selector.transform(data_x_cleaned), columns=data_x_cleaned.columns[selector.get_support()])
selected_features = data_x_cleaned.columns

# Grid search function
def gridsearch(gridlist, classlist, data_x, data_y, cv):
    best_estimators = []
    for i in range(len(gridlist)):
        grid = GridSearchCV(classlist[i], gridlist[i], scoring='roc_auc_ovr', cv=cv, n_jobs=-1, error_score='raise')
        grid.fit(data_x, data_y)
        best_estimators.append(grid.best_estimator_)
    return best_estimators

# Execute grid search
best_classlist = gridsearch(gridlist, classlist, data_x_cleaned, data_y_cleaned, cv)

# Cross-validation performance evaluation
metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
std_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
fold_metrics = []

for model, name in zip(best_classlist, ['SVM', 'RF', 'XGB']):
    fold_results = {'Fold': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
    for fold, (train_idx, val_idx) in enumerate(cv.split(data_x_cleaned, data_y_cleaned), 1):
        model.fit(data_x_cleaned.iloc[train_idx], data_y_cleaned.iloc[train_idx])
        pred = model.predict(data_x_cleaned.iloc[val_idx])
        y_score = model.predict_proba(data_x_cleaned.iloc[val_idx])
        acc = accuracy_score(data_y_cleaned.iloc[val_idx], pred)
        prec = precision_score(data_y_cleaned.iloc[val_idx], pred, average='weighted')
        rec = recall_score(data_y_cleaned.iloc[val_idx], pred, average='weighted')
        f1 = f1_score(data_y_cleaned.iloc[val_idx], pred, average='weighted')
        auc_score = roc_auc_score(label_binarize(data_y_cleaned.iloc[val_idx], classes=np.unique(data_y_cleaned)), y_score, multi_class='ovr')
        cm = confusion_matrix(data_y_cleaned.iloc[val_idx], pred)
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

print("Fold-wise Performance Metrics:")
print(all_fold_metrics)

print("\nPerformance Metrics on Cross-Validation:")
print(performance_metrics)

plt.figure(figsize=(10, 8))
colors = cycle(['#2ca02c', '#ff7f0e', '#1f77b4'])
line_styles = cycle(['-', '--', '-.', ':'])
models_auc_sorted = sorted(zip(best_classlist, ['RF', 'SVM', 'XGB'],
                               colors, line_styles,
                               performance_metrics['AUC'],
                               std_performance_metrics['AUC']),
                           key=lambda x: x[4], reverse=True)

for model, name, color, ls, mean_auc, std_auc in models_auc_sorted:
    y_score = cross_val_predict(model, data_x_cleaned, data_y_cleaned, cv=cv, method='predict_proba')
    fpr, tpr, _ = roc_curve(label_binarize(data_y_cleaned, classes=np.unique(data_y_cleaned)).ravel(), y_score.ravel())
    plt.plot(fpr, tpr, color=color, lw=2, linestyle=ls, 
             label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=22, fontname='Times New Roman')
plt.ylabel('True Positive Rate', fontsize=22, fontname='Times New Roman')
plt.legend(loc="lower right", fontsize=14)
plt.tight_layout()
plt.show()

# Plot confusion matrices for each model
fig, axs = plt.subplots(1, 3, figsize=(15, 7))
for ax, model, name in zip(axs.flatten(), best_classlist, ['SVM', 'RF', 'XGB']):
    all_preds, all_actuals = [], []
    for train_idx, test_idx in cv.split(data_x_cleaned, data_y_cleaned):
        model.fit(data_x_cleaned.iloc[train_idx], data_y_cleaned.iloc[train_idx])
        preds = model.predict(data_x_cleaned.iloc[test_idx])
        all_preds.extend(preds)
        all_actuals.extend(data_y_cleaned.iloc[test_idx])
    final_cm = confusion_matrix(all_actuals, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=final_cm)
    disp.plot(ax=ax, cmap='Greys', colorbar=False)
    ax.set_title(f'{name}', fontsize=22, fontname='Times New Roman')
    ax.set_xlabel('Predicted label', fontsize=22, fontname='Times New Roman')
    ax.set_ylabel('True label', fontsize=22, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=18)
    for text in ax.texts:
        text.set_fontsize(22)
        text.set_fontname('Times New Roman')
plt.tight_layout()
plt.show()

# Calculate and visualize feature importance
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Ran_state)
fold_feature_importances = {'RF': [], 'XGB': [], 'SVM': []}
for model, name in zip(best_classlist, ['SVM', 'RF', 'XGB']):
    for train_idx, val_idx in cv.split(data_x_cleaned, data_y_cleaned):
        model.fit(data_x_cleaned.iloc[train_idx], data_y_cleaned.iloc[train_idx])
        if name in ['RF', 'XGB']:
            importances = model.feature_importances_
        else:
            result = permutation_importance(model, data_x_cleaned.iloc[val_idx], data_y_cleaned.iloc[val_idx], 
                                            n_repeats=5, random_state=Ran_state, n_jobs=-1)
            importances = result.importances_mean
        fold_feature_importances[name].append(importances)

mean_feature_importances = {name: np.mean(importances, axis=0)
                            for name, importances in fold_feature_importances.items()}
feature_importances_df = pd.DataFrame(mean_feature_importances, index=selected_features)

# Plot average feature importance
mean_importances = feature_importances_df.mean(axis=1).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_importances.values, y=mean_importances.index, palette="Greys_r")
for i, v in enumerate(mean_importances):
    plt.text(v + 0.002, i, f'{v:.4f}', color='black', va='center', fontsize=22, fontname='Times New Roman')
plt.xticks(fontsize=22, fontname='Times New Roman')
plt.yticks(fontsize=22, fontname='Times New Roman')
for spine in ['top', 'right', 'left', 'bottom']:
    plt.gca().spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

# Visualize overall performance metrics
plt.figure(figsize=(10, 6))
sns.heatmap(performance_metrics.set_index('Model'), annot=True, fmt=".4f", cmap='Greys')
plt.title('Performance Metrics across Models (Mean)', fontsize=16, fontname='Times New Roman')
plt.tight_layout()
plt.show()

# Visualize standard deviation of performance
plt.figure(figsize=(10, 6))
sns.heatmap(std_performance_metrics, annot=True, fmt=".4f", cmap='Greys')
plt.title('Performance Metrics across Models (Standard Deviation)', fontsize=16, fontname='Times New Roman')
plt.tight_layout()
plt.show()

# Visualize full feature importance heatmap
plt.figure(figsize=(10, 6))  
sns.heatmap(feature_importances_df, annot=True, fmt=".4f", cmap='Greys')
plt.title('Feature Importance Matrix', fontsize=16, fontname='Times New Roman')
plt.tight_layout()
plt.show()

