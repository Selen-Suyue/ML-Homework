import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from termcolor import cprint

class CustomAdaBoost:

    def __init__(self, base_classifiers, iters=50):
        self.base_classifiers = base_classifiers
        self.iters = iters
        self.alphas = np.ones(len(base_classifiers)) / len(base_classifiers)
        self.models = []

    def fit(self, X, y):
        for estimator in self.base_classifiers:
            model = estimator.fit(X, y)
            self.models.append(model)
        N = len(y)
        w = np.ones(N) / N

        for i in range(self.iters):
            for t,model in enumerate(self.models):
                y_pred = model.predict(X)

                err = np.sum( w * (y_pred != y))
                alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

                w = w * np.exp(-alpha * y * y_pred)
                w = w / np.sum(w)

                self.alphas[t]=alpha

            self.alphas = self.alphas / np.sum(self.alphas)

    def predict(self, X):
        cprint(self.alphas,"light_cyan")
        y_pred = np.zeros(len(X))
        for model, alpha in zip(self.models, self.alphas):
            y_pred += alpha * model.predict(X)
        return np.round(y_pred)
    
def run(args):
    X1, y1 = make_classification(n_samples=1000, n_features=6, n_classes=2, random_state=args.rand)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=args.rand)

    dt_clf = DecisionTreeClassifier(max_depth=1)  
    svm_clf = SVC(kernel='sigmoid', probability=True) 
    lr_clf = LogisticRegression()  
    dt_clf.fit(X1_train, y1_train)
    svm_clf.fit(X1_train, y1_train)
    lr_clf.fit(X1_train, y1_train)

    y1_pred_dt_clf = dt_clf.predict(X1_test)
    y1_pred_svm_clf = svm_clf.predict(X1_test)
    y1_pred_lr_clf = lr_clf.predict(X1_test)
    

    acc_dt_clf = accuracy_score(y1_test, y1_pred_dt_clf)
    acc_svm_clf = accuracy_score(y1_test, y1_pred_svm_clf)
    acc_lr_clf = accuracy_score(y1_test, y1_pred_lr_clf)

    base_classifiers = [dt_clf, svm_clf, lr_clf]

   
    pca = PCA(n_components=2)
    X1_test_pca = pca.fit_transform(X1_test)
 
    plt.figure(figsize=(15, 10))
    plt.suptitle("PCA Visualization of Test Data")  

    plt.subplot(2, 3, 1)
    plt.scatter(X1_test_pca[:, 0], X1_test_pca[:, 1], c=y1_test, cmap='viridis', edgecolors='k')
    plt.title('True Labels')

    plt.subplot(2, 3, 2)
    plt.scatter(X1_test_pca[:, 0], X1_test_pca[:, 1], c=y1_pred_dt_clf, cmap='viridis', edgecolors='k')
    plt.title(f'Decision Tree (Acc: {acc_dt_clf:.2f})')

    plt.subplot(2, 3, 3)
    plt.scatter(X1_test_pca[:, 0], X1_test_pca[:, 1], c=y1_pred_svm_clf, cmap='viridis', edgecolors='k')
    plt.title(f'SVM (Acc: {acc_svm_clf:.2f})')

    plt.subplot(2, 3, 4)
    plt.scatter(X1_test_pca[:, 0], X1_test_pca[:, 1], c=y1_pred_lr_clf, cmap='viridis', edgecolors='k')
    plt.title(f'Logistic Regression (Acc: {acc_lr_clf:.2f})')
    
    for iter in range(1, 11):
        adaboost = CustomAdaBoost(base_classifiers=base_classifiers, iters=iter)
        adaboost.fit(X1_train, y1_train)
        y_pred_adaboost = adaboost.predict(X1_test)
        accuracy_adaboost = accuracy_score(y1_test, y_pred_adaboost)
        plt.subplot(2, 3, 5 if iter==1 else 6)
        plt.scatter(X1_test_pca[:, 0], X1_test_pca[:, 1], c=y_pred_adaboost, cmap='viridis', edgecolors='k')
        plt.title(f'Adaboost Iter {iter} (Acc: {accuracy_adaboost:.2f})')

        if iter == 1 :
            with open(f'results_iter_{iter}.txt', 'w') as f:
                f.write(f"#########Iter={iter}###########\n\n")
                f.write(f"Decision Tree  Accuracy: {acc_dt_clf}\n")
                f.write(f"SVM Accuracy: {acc_svm_clf}\n")
                f.write(f"Logistic Regression Accuracy: {acc_lr_clf}\n\n")        
                f.write(f"Custom AdaBoost Accuracy: {accuracy_adaboost}\n\n\n")
        else:
            with open(f'results_iter_{iter}.txt', 'w') as f:
                f.write(f"#########Iter={iter}###########\n")
                f.write(f"Custom AdaBoost Accuracy: {accuracy_adaboost}\n\n\n")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.savefig('pca_visualization_comparison.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaboost")
    parser.add_argument("--rand", type=int, default=7, help="")

    args = parser.parse_args()
    run(args)