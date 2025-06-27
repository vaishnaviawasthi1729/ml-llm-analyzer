import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import roc_curve,auc, ConfusionMatrixDisplay
import os 
import time 
#pip install matplotlib
def extract_supported_models(llm_text):
    # Remove asterisks and lowercase everything for easier matching
    clean_text = llm_text.replace('*', '').replace('\n', ' ').lower()
    allowed = [
        ('Logistic Regression', r'logistic regression'),
        ('Decision Tree', r'decision tree(\s*classifier)?'),
        ('Random Forest', r'random forest(\s*classifier)?'),
        ('Linear Regression', r'linear regression'),
        ('Support Vector Machine', r'support vector machine(\s*classifier)?'),
        ('K-Nearest Neighbors', r'k-nearest neighbors(\s*classifier)?'),
        ('Naive Bayes', r'naive bayes(\s*classifier)?'),
        ('XGBoost', r'xgboost(\s*classifier)?')  
    ]
    models = []
    for model, pattern in allowed:
        if re.search(pattern, clean_text, re.IGNORECASE):
            models.append(model)
    return models
def save_confusion_matrix(y_true, y_pred, model_name):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    filename=f"{int(time.time())}_{model_name.replace(' ', '')}"
    filepath=f'static/images/conf_{filename}.png'
    plt.savefig(filepath)
    plt.close()
    return filepath
def save_roc_curve(y_true, y_scores, model_name):
    if len(set(y_true)) <= 2:
        fpr,tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,tpr,label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing' )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')   
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')   
        plt.tight_layout()
        filename=f"{int(time.time())}_{model_name.replace(' ', '')}"
        filepath=f'static/images/roc_{filename}.png'
        plt.savefig(filepath)
        plt.close()
        return filepath
    else:   
        return None
def average_score(m):
    if m.get('type')=='classification':
        valid_metrics=[m["accuracy"],m["precision"],m["recall"],m["f1_score"]] 
        if m["auc"]: valid_metrics.append(m["auc"])
        return sum(valid_metrics)/len(valid_metrics) if valid_metrics else 0
    elif m.get('type')=='regression':
        return m.get('r2',0)
    return 0
