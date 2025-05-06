import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 50
CV_SPLITS = 5

# Define models and hyperparameter grids
models = {
    'SVM': SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE),
    'RF': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    'XGB': XGBClassifier(objective='multi:softprob', random_state=RANDOM_STATE, n_jobs=-1)
}

grids = {
    'SVM': {
        'C': [0.1, 10, 80],
        'gamma': ['scale', 0.1, 1, 50],
        'kernel': ['rbf']
    },
    'RF': {
        'n_estimators': [50, 100, 200],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.7],
        'bootstrap': [True],
        'max_depth': [None, 5, 10, 20, 30],
        'class_weight': ['balanced']
    },
    'XGB': {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1, 0.3, 0.8],
        'max_depth': [7, 8],
        'subsample': [0.3, 0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1]
    }
}

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Sex'] = df['Sex'].map({'남': 0, '여': 1, 'male': 0, 'female': 1})
    y = df['target']
    X = df.drop(columns=['target'])

    iso_forest = IsolationForest(contamination=0.01, random_state=RANDOM_STATE, n_jobs=-1)
    outliers_pred = iso_forest.fit_predict(X)
    X_cleaned = X.iloc[outliers_pred == 1]
    y_cleaned = y.iloc[outliers_pred == 1]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_cleaned), columns=X_cleaned.columns)
    return X_scaled, y_cleaned

def evaluate_model(X, y, model_key, grid):
    model = models[model_key]
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(model, grid, scoring='f1_weighted', cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    scores = cross_validate(best_model, X, y, cv=cv, scoring=[
        'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr'
    ], return_train_score=False)

    return {
        'Model': model_key,
        'Accuracy': np.mean(scores['test_accuracy']),
        'Precision': np.mean(scores['test_precision_weighted']),
        'Recall': np.mean(scores['test_recall_weighted']),
        'F1-score': np.mean(scores['test_f1_weighted']),
        'AUC': np.mean(scores['test_roc_auc_ovr']),
        'Accuracy_std': np.std(scores['test_accuracy']),
        'F1-score_std': np.std(scores['test_f1_weighted'])
    }

def run_lasso_selection(X, y):
    print("\n[INFO] Running LASSO feature selection...")
    lasso = LogisticRegression(penalty='l1', solver='saga', C=0.01, multi_class='ovr', max_iter=5000, random_state=RANDOM_STATE)
    selector = SelectFromModel(estimator=lasso)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]
    print(f"[DEBUG] LASSO selected features: {list(selected)}")
    return X[selected], selected

def run_ridge_selection(X, y):
    print("\n[INFO] Running RIDGE feature selection...")
    ridge = LogisticRegression(penalty='l2', solver='saga', C=1.0, multi_class='ovr', max_iter=5000, random_state=RANDOM_STATE)
    selector = SelectFromModel(estimator=ridge, threshold='median')
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]
    if len(selected) == 0:
        raise ValueError("No features selected by RIDGE. Adjust threshold or parameters.")
    print(f"[DEBUG] RIDGE selected features: {list(selected)}")
    return X[selected], selected

def run_rfe_selection(X, y):
    print("\n[INFO] Running RFE feature selection...")
    base = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=RANDOM_STATE)
    rfe = RFE(estimator=base, n_features_to_select=8)
    rfe.fit(X, y)
    selected = X.columns[rfe.get_support()]
    print(f"[DEBUG] RFE selected features: {list(selected)}")
    return X[selected], selected

def run_feature_selection_comparison(file_path):
    X, y = preprocess_data(file_path)
    results = []

    for method, selector_func in zip(['LASSO', 'RIDGE', 'RFE'], [run_lasso_selection, run_ridge_selection, run_rfe_selection]):
        X_sel, _ = selector_func(X, y)
        for model_key in models:
            print(f"[INFO] Evaluating {method} + {model_key}...")
            result = evaluate_model(X_sel, y, model_key, grids[model_key])
            result['FeatureSelection'] = method
            results.append(result)

    df_results = pd.DataFrame(results)
    print("\n[RESULTS] Feature Selection Comparison:")
    print(df_results)
    df_results.to_csv("./results/feature_selection_comparison.csv", index=False)

if __name__ == "__main__":
    run_feature_selection_comparison("./data/preprocessed_dataset.csv")
