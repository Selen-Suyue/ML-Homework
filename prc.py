def precision_recall_curve(y_true, y_scores):
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    y_true = [y_true[i] for i in sorted_indices]
    y_scores = [y_scores[i] for i in sorted_indices]
    
    tp = 0  
    fp = 0  
    fn = sum(y_true)  
    precision, recall = [], []
    thresholds = []
    
    for i, score in enumerate(y_scores):
        if y_true[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        thresholds.append(score)
    
    return precision, recall, thresholds

def auc(x, y):
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    return area

import matplotlib.pyplot as plt

def plot_prc_curve(recall, precision, auc_score):
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'PRC (AUC={auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()