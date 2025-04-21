import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, StratifiedKFold, cross_val_predict
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

# 데이터 로드
file_path = r"C:\Users\sg010\Desktop\wlsWkdlqslek.csv"
df = pd.read_csv(file_path)

# 'Sex' 열 인코딩
df['Sex'] = df['Sex'].map({'남': 0, '여': 1})

# 타겟과 특징 정의
data_y = df['target']
data_x = df.drop('target', axis=1)

# Isolation Forest로 이상치 제거
iso_forest = IsolationForest(contamination=0.01, random_state=50, n_jobs=-1)
outliers_pred = iso_forest.fit_predict(data_x)

# 이상치 제거
data_x_cleaned = data_x.iloc[outliers_pred == 1]
data_y_cleaned = data_y.iloc[outliers_pred == 1]

# 특징 스케일링
scaler = MinMaxScaler()
data_x_scaled = scaler.fit_transform(data_x_cleaned)
data_x_cleaned = pd.DataFrame(data_x_scaled, columns=data_x_cleaned.columns)

# 모델 및 하이퍼파라미터 정의
Ran_state = 50
RF = RandomForestClassifier(class_weight='balanced', random_state=Ran_state, n_jobs=-1)
XGB = XGBClassifier(objective='multi:softprob', random_state=Ran_state, n_jobs=-1)
SVM = SVC(class_weight='balanced', probability=True, random_state=Ran_state)

# Random Forest 하이퍼파라미터 그리드
SVM_grid = {
    'C': [80],
    'gamma': ['scale', 0.1, 1, 50],
    'kernel': ['rbf'],
    'degree': [2, 3]  # poly 커널에 사용
}

RF_grid = {
    'n_estimators': [98, 102],  # 트리의 개수를 50, 75, 100으로 설정하여 작은 데이터셋에 적합하게 설정  #100
    'min_samples_leaf': [1, 2, 4],  # 리프 노드의 최소 샘플 수를 줄여 세밀한 패턴을 학습
    'max_features': ['sqrt', 0.7],  # 피처 사용 비율을 70%로 추가해 다양한 피처를 활용
    'bootstrap': [True],  # 부트스트랩 샘플링 사용
    'max_depth': [None, 5, 10, 20, 30],  # 트리 깊이를 제한하지 않거나 20, 30으로 설정
    'class_weight': ['balanced'],  # 클래스 불균형에 대한 가중치 적용
}

# XGBoost 하이퍼파라미터 그리드
XGB_grid = {
    'n_estimators': [100, 300],  # 트리 개수
    'learning_rate': [0.8],  # 학습률
    'max_depth': [7, 8],  # 트리의 최대 길이
    'subsample': [0.3, 0.8, 1.0],  # 각 트리의 사벌린 비율
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],  # 리프 노드 분할에 필요한 최소 손실 감소
}

gridlist = [SVM_grid, RF_grid, XGB_grid]
classlist = [SVM, RF, XGB]

# k값 설정
k = 5  # 원하는 fold 수를 설정

# GridSearch 함수 정의
def gridsearch(gridlist, classlist, data_x, data_y, cv):
    best_estimators = []
    for i in range(len(gridlist)):
        grid = GridSearchCV(classlist[i], gridlist[i], scoring='roc_auc_ovr', cv=cv, n_jobs=-1, error_score='raise')
        grid.fit(data_x, data_y)
        best_estimators.append(grid.best_estimator_)
    return best_estimators

# StratifiedKFold 설정
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=Ran_state)

# 특징 선택 (RandomForestClassifier 사용)
rf_selector = RandomForestClassifier(n_estimators=200, random_state=Ran_state, n_jobs=-1)
rf_selector.fit(data_x_cleaned, data_y_cleaned)
selector = SelectFromModel(rf_selector, prefit=True)

# SelectFromModel에 데이터프레임 전달하여 feature names 유지
data_x_cleaned = pd.DataFrame(selector.transform(data_x_cleaned), columns=data_x_cleaned.columns[selector.get_support()])

selected_features = data_x_cleaned.columns  # 선택된 특징 목록

# GridSearchCV 실행
best_classlist = gridsearch(gridlist, classlist, data_x_cleaned, data_y_cleaned, cv)

# 성능 지표 및 AUC의 평균과 표준편차 계산
metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
std_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}

# 각 fold의 성능 지표를 저장할 데이터프레임 생성
fold_metrics = []

# 각 모델별로 K-Fold 교차 검증 수행
for model, name in zip(best_classlist, ['SVM', 'RF', 'XGB']):
    fold_results = {'Fold': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'TPR': [], 'TNR': []}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(data_x_cleaned, data_y_cleaned), 1):
        model.fit(data_x_cleaned.iloc[train_idx], data_y_cleaned.iloc[train_idx])
        pred = model.predict(data_x_cleaned.iloc[val_idx])
        y_score = model.predict_proba(data_x_cleaned.iloc[val_idx])
        
        # 각 성능 지표 계산
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
        
        TPR = TP / (TP + FN)  # 민감도 (Sensitivity)
        TNR = TN / (TN + FP)  # 특이도 (Specificity)
        
        # 평균 TPR, TNR 계산
        avg_TPR = np.mean(TPR)
        avg_TNR = np.mean(TNR)
        
        # fold 별 성능 지표 저장
        fold_results['Fold'].append(f'Fold {fold}')
        fold_results['Accuracy'].append(acc)
        fold_results['Precision'].append(prec)
        fold_results['Recall'].append(rec)
        fold_results['F1-score'].append(f1)
        fold_results['AUC'].append(auc_score)
        fold_results['TPR'].append(avg_TPR)
        fold_results['TNR'].append(avg_TNR)
    
    # fold 별 결과를 데이터프레임으로 변환하여 fold_metrics 리스트에 저장
    fold_df = pd.DataFrame(fold_results)
    fold_df['Model'] = name
    fold_metrics.append(fold_df)

# 모든 fold 결과를 합친 데이터프레임
all_fold_metrics = pd.concat(fold_metrics)

# 각 모델에 대한 평균 및 표준편차 계산
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

# 결과 정리 및 출력
performance_metrics = pd.DataFrame(metrics)
std_performance_metrics = pd.DataFrame(std_metrics, index=performance_metrics['Model'])

# 각 fold의 성능 지표 출력
print("Fold-wise Performance Metrics:")
print(all_fold_metrics)

print("\nPerformance Metrics on Cross-Validation:")
print(performance_metrics)

# ROC 곡선 시각화 (AUC 값이 높은 순서대로 정렬, 평균 및 표준편차 포함)
plt.figure(figsize=(10, 8))
colors = cycle(['#2ca02c', '#ff7f0e', '#1f77b4'])  # 색상 설정
line_styles = cycle(['-', '--', '-.', ':'])  # 선 스타일 설정

# AUC 평균 및 표준편차 계산 (AUC 값에 따른 모델 순서 정렬)
# AUC 값에 따라 모델의 순서가 달라지도록 설정
models_auc_sorted = sorted(zip(best_classlist, 
                               ['RF', 'SVM', 'XGB'], 
                               colors, 
                               line_styles, 
                               performance_metrics['AUC'], 
                               std_performance_metrics['AUC']),
                           key=lambda x: x[4], reverse=True)  # AUC 값에 따라 정렬

# 정렬된 순서대로 ROC 곡선을 그림
for model, name, color, ls, mean_auc, std_auc in models_auc_sorted:
    y_score = cross_val_predict(model, data_x_cleaned, data_y_cleaned, cv=cv, method='predict_proba')
    fpr, tpr, _ = roc_curve(label_binarize(data_y_cleaned, classes=np.unique(data_y_cleaned)).ravel(), y_score.ravel())
    plt.plot(fpr, tpr, color=color, lw=2, linestyle=ls, 
             label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')  # AUC 값과 표준편차를 레이블에 정확히 표시

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False-positive rate', fontsize=22, fontname='Times New Roman')
plt.ylabel('True-positive rate', fontsize=22, fontname='Times New Roman')
plt.title('', fontsize=18, fontname='Times New Roman')
plt.legend(loc="lower right", fontsize=14)
plt.show()

# 평균 혼동 행렬 시각화
fig, axs = plt.subplots(1, 3, figsize=(15, 7))

# 각 모델에 대한 혼동 행렬 계산
for ax, model, name in zip(axs.flatten(), best_classlist, ['SVM', 'RF', 'XGB']):
    # 모든 fold의 예측값과 실제값을 수집하기 위한 리스트
    all_preds = []
    all_actuals = []

    # StratifiedKFold 정의
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=Ran_state)
    
    for train_idx, test_idx in cv.split(data_x_cleaned, data_y_cleaned):
        # 모델 학습
        model.fit(data_x_cleaned.iloc[train_idx], data_y_cleaned.iloc[train_idx])
        # 검증 데이터에 대한 예측값 수집
        preds = model.predict(data_x_cleaned.iloc[test_idx])
        # 예측값과 실제값 수집
        all_preds.extend(preds)
        all_actuals.extend(data_y_cleaned.iloc[test_idx])

    # 수집된 모든 예측값과 실제값을 기반으로 최종 혼동 행렬 계산
    final_cm = confusion_matrix(all_actuals, all_preds)

    # 혼동 행렬 시각화
    disp = ConfusionMatrixDisplay(confusion_matrix=final_cm)
    disp.plot(ax=ax, cmap='Greys', colorbar=False)
    ax.set_title(f'Class {["1", "2", "3"][axs.tolist().index(ax)]}', fontsize=22, fontname='Times New Roman')

    # X축과 Y축에 텍스트 추가 및 스타일 설정
    ax.set_xlabel('True label', fontsize=22, fontname='Times New Roman')
    ax.set_ylabel('Predicted label', fontsize=22, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=18)  # x축과 y축 글씨 크기 조정
    for text in ax.texts:  # 혼동 행렬의 숫자 글씨 크기 조정
        text.set_fontsize(22)
        text.set_fontname('Times New Roman')

plt.tight_layout()
plt.show()

# 5-fold 교차검증을 통한 특징 중요도 계산 및 시각화
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Ran_state)

# 모델별 특징 중요도를 저장할 리스트
fold_feature_importances = {'RF': [], 'XGB': [], 'SVM': []}

# 각 모델별로 교차검증을 수행하면서 특징 중요도 계산
for model, name in zip(best_classlist, ['SVM', 'RF', 'XGB']):
    for train_idx, val_idx in cv.split(data_x_cleaned, data_y_cleaned):
        # 모델 학습
        model.fit(data_x_cleaned.iloc[train_idx], data_y_cleaned.iloc[train_idx])
        
        if name in ['RF', 'XGB']:
            # RandomForest와 XGBoost의 feature_importances_ 사용
            importances = model.feature_importances_
        else:
            # SVM의 경우 permutation_importance 사용
            result = permutation_importance(model, data_x_cleaned.iloc[val_idx], data_y_cleaned.iloc[val_idx], 
                                            n_repeats=5, random_state=Ran_state, n_jobs=-1)
            importances = result.importances_mean
        
        # 현재 fold의 특징 중요도 저장
        fold_feature_importances[name].append(importances)

# 각 모델별로 fold들의 특징 중요도를 평균내어 최종 중요도를 계산 
mean_feature_importances = {}
for name in fold_feature_importances:
    mean_feature_importances[name] = np.mean(fold_feature_importances[name], axis=0)

# 특징 중요도 시각화
feature_importances_df = pd.DataFrame(mean_feature_importances, index=selected_features)

# 각 모델별로 fold들의 특징 중요도를 평균내어 최종 중요도를 계산 
mean_feature_importances = {}
for name in fold_feature_importances:
    mean_feature_importances[name] = np.mean(fold_feature_importances[name], axis=0)

# 특징 중요도 시각화
feature_importances_df = pd.DataFrame(mean_feature_importances, index=selected_features)

# 모든 모델의 중요도 평균을 구하고 높은 순서대로 정렬하여 막대그래프로 시각화
mean_importances = feature_importances_df.mean(axis=1).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=mean_importances.values, y=mean_importances.index, palette="Greys_r")

# 막대 옆에 정확한 수치를 표시 (위치를 살짝 조정)
for i, v in enumerate(mean_importances):
    plt.text(v + 0.002, i, f'{v:.4f}', color='black', va='center', fontsize=22, fontname='Times New Roman')  # 글씨 크기와 폰트 조정

plt.title('', fontsize=26, fontname='Times New Roman')
plt.xticks(fontsize=22, fontname='Times New Roman')  # x축 글씨 크기와 폰트 조정
plt.yticks(fontsize=22, fontname='Times New Roman')  # y축 글씨 크기와 폰트 조정
plt.gca().spines['top'].set_visible(False)  # 상단 테두리 제거
plt.gca().spines['right'].set_visible(False)  # 오른쪽 테두리 제거
plt.gca().spines['left'].set_visible(False)  # 왼쪽 테두리 제거
plt.gca().spines['bottom'].set_visible(False)  # 아래쪽 테두리 제거
plt.show()

# 성능 지표 시각화 (평균)
plt.figure(figsize=(10, 6))
sns.heatmap(performance_metrics.set_index('Model'), annot=True, fmt=".4f", cmap='Greys')
plt.title('Performance Metrics across Models (Mean)', fontsize=16, fontname='Times New Roman')
plt.show()

# 성능 지표 시각화 (표준편차)
plt.figure(figsize=(10, 6))
sns.heatmap(std_performance_metrics, annot=True, fmt=".4f", cmap='Greys')
plt.title('Performance Metrics across Models (Standard Deviation)', fontsize=16, fontname='Times New Roman')
plt.show()

# 특징 중요도 행렬 시각화
plt.figure(figsize=(10, 6))  
sns.heatmap(feature_importances_df, annot=True, fmt=".4f", cmap='Greys')
plt.title('', fontsize=22, fontname='Times New Roman')
plt.show()

