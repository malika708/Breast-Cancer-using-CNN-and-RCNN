import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics dynamically for any number of classes."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    }
    cm = confusion_matrix(y_true, y_pred)
    # Specificity: mean of TN/(TN+FP) for all classes
    specificity_list = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    metrics['specificity'] = np.mean(specificity_list) * 100
    return metrics, cm

def plot_confusion_matrix(cm, labels, title):
    """Plot confusion matrix heatmap for any number/order of classes."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
                annot=True, cmap="viridis", fmt="g", annot_kws={"size": 16})
    plt.title(title)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def plot_training_history(history, optimizer_name):
    """Plot training and validation accuracy/loss curves."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'{optimizer_name} Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{optimizer_name} Model Loss')
    
    plt.tight_layout()
    plt.show()

def plot_comparison_graph(*optim_metrics):
    """Plot comparison bar graph for any number of optimizers and metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    barWidth = 0.8 / len(optim_metrics)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    r = np.arange(len(metrics))
    for i, (optim_dict, label) in enumerate(optim_metrics):
        values = [np.mean(optim_dict[metric]) for metric in metrics]
        ax1.bar(r + i * barWidth, values, width=barWidth, edgecolor='grey', label=label)
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_xticks(r + barWidth * (len(optim_metrics) - 1) / 2)
    ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'])
    ax1.set_title('Comparison of Optimizers')
    ax1.legend()
    plt.show() 