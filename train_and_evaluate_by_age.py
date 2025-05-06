# 5. Model Training and Evaluation by Age Group

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# Load dataset
file_path = r"C:\Users\sg010\Desktop\wlsWkdlqslek.csv"
df = pd.read_csv(file_path)
df['Sex'] = df['Sex'].map({'남': 0, '여': 1})

# Split by age group
under_60_group = df[df['Age'] < 60]
over_60_group = df[df['Age'] >= 60]

# Choose group (change here as needed)
age_group_x = over_60_group.drop(['target', 'Age'], axis=1)
age_group_y = over_60_group['target']

# Remove outliers
iso_forest = IsolationForest(contamination=0.01, random_state=50, n_jobs=-1)
outliers_pred = iso_forest.fit_predict(age_group_x)
age_group_x = age_group_x.iloc[outliers_pred == 1]
age_group_y = age_group_y.iloc[outliers_pred == 1]

# Feature scaling
scaler = MinMaxScaler()
age_group_x_scaled = scaler.fit_transform(age_group_x)
age_group_x = pd.DataFrame(age_group_x_scaled, columns=age_group_x.columns)

# Feature selection
rf_selector = RandomForestClassifier(n_estimators=200, random_state=50, n_jobs=-1)
rf_selector.fit(age_group_x, age_group_y)
selector = SelectFromModel(rf_selector, prefit=True)
age_group_x = pd.DataFrame(selector.transform(age_group_x),
                            columns=age_group_x.columns[selector.get_support()])
selected_features = age_group_x.columns

# Cross-validation setting
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

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

models = [SVC(class_weight='balanced', probability=True, random_state=50),
          RandomForestClassifier(class_weight='balanced', random_state=50, n_jobs=-1),
          XGBClassifier(objective='multi:softprob', random_state=50, n_jobs=-1)]
grids = [SVM_grid, RF_grid, XGB_grid]
model_names = ['SVM', 'RF', 'XGB']

# Grid search
best_models = []
for model, grid in zip(models, grids):
    gs = GridSearchCV(model, grid, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)
    gs.fit(age_group_x, age_group_y)
    best_models.append(gs.best_estimator_)

# Cross-validated evaluation
results = []
for model, name in zip(best_models, model_names):
    fold_scores = {'Model': name}
    accs, precs, recs, f1s, aucs, tprs, tnrs = [], [], [], [], [], [], []

    for train_idx, val_idx in cv.split(age_group_x, age_group_y):
        X_train, X_val = age_group_x.iloc[train_idx], age_group_x.iloc[val_idx]
        y_train, y_val = age_group_y.iloc[train_idx], age_group_y.iloc[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        accs.append(accuracy_score(y_val, preds))
        precs.append(precision_score(y_val, preds, average='weighted'))
        recs.append(recall_score(y_val, preds, average='weighted'))
        f1s.append(f1_score(y_val, preds, average='weighted'))
        aucs.append(roc_auc_score(label_binarize(y_val, classes=np.unique(age_group_y)), probs, multi_class='ovr'))

        cm = confusion_matrix(y_val, preds)
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        tprs.append(np.mean(TPR))
        tnrs.append(np.mean(TNR))

    fold_scores['Accuracy'] = np.mean(accs)
    fold_scores['Precision'] = np.mean(precs)
    fold_scores['Recall'] = np.mean(recs)
    fold_scores['F1-score'] = np.mean(f1s)
    fold_scores['AUC'] = np.mean(aucs)
    fold_scores['TPR'] = np.mean(tprs)
    fold_scores['TNR'] = np.mean(tnrs)

    results.append(fold_scores)

performance_df = pd.DataFrame(results)
print(performance_df)

# Visualization: Performance heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(performance_df.set_index('Model'), annot=True, fmt=".4f", cmap='Greys')
plt.title('Performance Metrics across Models')
plt.show()

# Feature importance
fold_feature_importances = {name: [] for name in model_names}
for model, name in zip(best_models, model_names):
    for train_idx, val_idx in cv.split(age_group_x, age_group_y):
        model.fit(age_group_x.iloc[train_idx], age_group_y.iloc[train_idx])
        if name in ['RF', 'XGB']:
            importances = model.feature_importances_
        else:
            result = permutation_importance(model, age_group_x.iloc[val_idx], age_group_y.iloc[val_idx],
                                            n_repeats=5, random_state=50, n_jobs=-1)
            importances = result.importances_mean
        fold_feature_importances[name].append(importances)

mean_importances = {name: np.mean(fold_feature_importances[name], axis=0) for name in fold_feature_importances}
feature_importances_df = pd.DataFrame(mean_importances, index=selected_features)

plt.figure(figsize=(12, 6))
sns.heatmap(feature_importances_df, annot=True, fmt=".4f", cmap="Greys")
plt.title("Feature Importances across Models")
plt.show()
