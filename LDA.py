import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, accuracy_score, auc
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

datasets = [
    ("Breast Cancer", load_breast_cancer()),
    ("Iris (Binary Subset)", load_iris()),  # 取前两类作为二分类
    ("Wine (Binary Subset)", load_wine())   # 取前两类作为二分类
]

results = []


for name, dataset in datasets:
    X, y = dataset.data, dataset.target
    
    if name != "Breast Cancer":
        X = X[y != 2]
        y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    

    lda = LDA()
    lda.fit(X_train, y_train)
    y_scores = lda.predict_proba(X_test)[:, 1]
    y_pred = lda.predict(X_test)
    

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    prc_auc = auc(recalls, precisions)

    results.append((name, accuracy, precision, recall, prc_auc))
    
    plt.figure()
    plt.plot(recalls, precisions, label=f'{name} (AUC={prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {name}')
    plt.legend()
    plt.grid()
    plt.show()

for name, accuracy, precision, recall, prc_auc in results:
    print(f"Dataset: {name}")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  PRC AUC: {prc_auc:.2f}")
