import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (roc_curve, auc, roc_auc_score, accuracy_score, 
                             precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
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

# Split dataset by sex
df_male = df[df['Sex'] == 0].copy()
df_female = df[df['Sex'] == 1].copy()

# Define target and features for both groups
def prepare_group(df_group):
    data_y = df_group['target']
    data_x = df_group.drop('target', axis=1)

    # Outlier removal
    iso_forest = IsolationForest(contamination=0.01, random_state=50, n_jobs=-1)
    outliers_pred = iso_forest.fit_predict(data_x)
    data_x_cleaned = data_x.iloc[outliers_pred == 1]
    data_y_cleaned = data_y.iloc[outliers_pred == 1]

    # Feature scaling
    scaler = MinMaxScaler()
    data_x_scaled = scaler.fit_transform(data_x_cleaned)
    data_x_cleaned = pd.DataFrame(data_x_scaled, columns=data_x_cleaned.columns)

    # Feature selection
    rf_selector = RandomForestClassifier(n_estimators=200, random_state=50, n_jobs=-1)
    rf_selector.fit(data_x_cleaned, data_y_cleaned)
    selector = SelectFromModel(rf_selector, prefit=True)
    selected_x = pd.DataFrame(selector.transform(data_x_cleaned), columns=data_x_cleaned.columns[selector.get_support()])

    return selected_x, data_y_cleaned, selected_x.columns

# Define models and hyperparameter grids
Ran_state = 50
k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=Ran_state)

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

def gridsearch(gridlist, classlist, data_x, data_y, cv):
    best_estimators = []
    for grid, clf in zip(gridlist, classlist):
        grid_search = GridSearchCV(clf, grid, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)
        grid_search.fit(data_x, data_y)
        best_estimators.append(grid_search.best_estimator_)
    return best_estimators

def evaluate_group(data_x, data_y, selected_features):
    best_classlist = gridsearch([SVM_grid, RF_grid, XGB_grid], [SVM, RF, XGB], data_x, data_y, cv)

    metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
    std_metrics = {key: [] for key in list(metrics.keys())[1:]}
    fold_metrics = []

    for model, name in zip(best_classlist, ['SVM', 'RF', 'XGB']):
        fold_results = {key: [] for key in metrics.keys() if key != 'Model'}
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
            fold_results['Accuracy'].append(acc)
            fold_results['Precision'].append(prec)
            fold_results['Recall'].append(rec)
            fold_results['F1-score'].append(f1)
            fold_results['AUC'].append(auc_score)
            fold_results['TPR'].append(np.mean(TPR))
            fold_results['TNR'].append(np.mean(TNR))

        for key in fold_results:
            metrics[key].append(np.mean(fold_results[key]))
            std_metrics[key].append(np.std(fold_results[key]))
        metrics['Model'].append(name)

    performance_metrics = pd.DataFrame(metrics)
    std_performance_metrics = pd.DataFrame(std_metrics, index=performance_metrics['Model'])
    return performance_metrics, std_performance_metrics, best_classlist

# Run for male group
male_x, male_y, male_features = prepare_group(df_male)
male_perf, male_std, male_models = evaluate_group(male_x, male_y, male_features)

# Run for female group
female_x, female_y, female_features = prepare_group(df_female)
female_perf, female_std, female_models = evaluate_group(female_x, female_y, female_features)

print("Male Performance Metrics:\n", male_perf)
print("\nFemale Performance Metrics:\n", female_perf)
