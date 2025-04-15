import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch

from data_loader import load_fold, get_conditions_by_group, GROUPS

def analyze_data_distribution(fold, language, group):
    """Analyze and print the data distribution across mental health conditions"""
    print(f"\n{'='*60}")
    print(f"DATA DISTRIBUTION ANALYSIS (Fold {fold}, Language: {language}, Group: {group})")
    print(f"{'='*60}")
    
    # Get conditions for the specified group
    conditions = get_conditions_by_group(group)
    
    # Load dataset for this fold
    dataset_dict = load_fold(fold, language, group)
    
    # Analyze train dataset
    train_dataset = dataset_dict["train"]
    print("\nTRAINING SET DISTRIBUTION:")
    analyze_condition_distribution(train_dataset, conditions)
    
    # Analyze test dataset
    test_dataset = dataset_dict["test"] 
    print("\nTESTING SET DISTRIBUTION:")
    analyze_condition_distribution(test_dataset, conditions)

def analyze_condition_distribution(dataset, conditions):
    """Calculate and print distribution statistics for a dataset"""
    # Count samples per condition
    condition_counts = {}
    for condition in conditions:
        count = sum(1 for item in dataset if item["class"] == condition)
        condition_counts[condition] = count
    
    # Calculate total
    total_samples = sum(condition_counts.values())
    
    # Print distribution
    print(f"Total samples: {total_samples}")
    print(f"{'Condition':<15} | {'Count':<8} | {'Percentage':<10}")
    print(f"{'-'*15} | {'-'*8} | {'-'*10}")
    
    for condition, count in condition_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{condition:<15} | {count:<8} | {percentage:.2f}%")
        
def compute_class_weights(dataset, num_labels):
    """
    Compute class weights inversely proportional to class frequencies
    """
    # Count occurrences of each label
    label_counts = {}
    for i in range(num_labels):
        label_counts[i] = 0
    
    for example in dataset:
        label = example["label"].item()
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Calculate weights (inverse of frequency)
    total_samples = len(dataset)
    class_weights = {}
    for label, count in label_counts.items():
        class_weights[label] = total_samples / (count * num_labels)
    
    return class_weights

def validate_class_weights(dataset, num_labels):
    """Print detailed class distribution and weights"""
    # Count occurrences of each label
    label_counts = {}
    for i in range(num_labels):
        label_counts[i] = 0
    
    for example in dataset:
        label = example["label"].item()
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Calculate total
    total_samples = len(dataset)
    
    # Calculate and print weights
    print("\nClass distribution and weights:")
    print(f"{'Label':<6} | {'Count':<8} | {'Percentage':<10} | {'Weight':<8}")
    print(f"{'-'*6} | {'-'*8} | {'-'*10} | {'-'*8}")
    
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_samples) * 100
        weight = total_samples / (count * num_labels)
        print(f"{label:<6} | {count:<8} | {percentage:.2f}% | {weight:.4f}")
    
    return {label: total_samples / (count * num_labels) for label, count in label_counts.items()}

def compute_metrics(pred):
    """Compute evaluation metrics for the model"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze mental health tweets data')
    parser.add_argument('--fold', type=int, default=1, help='Fold number to analyze')
    parser.add_argument('--language', type=str, default='eng', choices=['eng', 'esp'], 
                        help='Language of the dataset')
    parser.add_argument('--group', type=str, default='all', choices=list(GROUPS.keys()),
                        help='Mental health group to analyze')
    
    args = parser.parse_args()
    
    analyze_data_distribution(args.fold, args.language, args.group)