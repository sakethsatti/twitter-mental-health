import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datetime import datetime

from data_loader import load_fold, get_conditions_by_group, GROUPS
from data_tests import (
    compute_metrics, 
    plot_confusion_matrix,
)
from weighted_trainer import WeightedTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a BERT model on mental health tweets')
    parser.add_argument('--model_name', type=str, default='vinai/bertweet-base',
                        help='Pretrained model name')
    parser.add_argument('--language', type=str, default='eng', choices=['eng', 'esp'],
                        help='Language of the dataset')
    parser.add_argument('--group', type=str, default='all', 
                        choices=list(GROUPS.keys()),
                        help='Mental health group to analyze')
    parser.add_argument('--folds', type=str, default='1,2,3,4,5',
                        help='Comma-separated list of folds to train on')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=96,
                        help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--weighted_training', action='store_true', default=False,
                    help='Use weighted training to handle class imbalance (enabled by default)')
    parser.add_argument('--balance_level', type=str, default='full', 
                        choices=['none', 'partial', 'full'],
                        help='''Data balancing strategy: 
                                "none": No balancing, 
                                "partial": Majority class reduced to size of largest minority class, 
                                "full": All classes reduced to size of smallest class''')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for hidden and attention layers')
    return parser.parse_args()

def preprocess_dataset(dataset, tokenizer, max_length, text_column="tweet", conditions=None):
    if text_column not in dataset.column_names:
        available_columns = dataset.column_names
        potential_text_columns = [col for col in available_columns if 'text' in col.lower() or 'tweet' in col.lower()]
        if potential_text_columns:
            text_column = potential_text_columns[0]
            print(f"Using {text_column} as the text column")
        else:
            raise ValueError(f"Text column not found in dataset. Available columns: {available_columns}")
    label_mapping = {condition: idx for idx, condition in enumerate(conditions)}
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        tokenized["label"] = [label_mapping[condition] for condition in examples["class"]]
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != "class"]
    )
    
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset, label_mapping

def train_fold(fold, args):
    print(f"\n{'='*50}")
    print(f"Training on Fold {fold}")
    print(f"{'='*50}")

    fold_output_dir = os.path.join(args.output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    conditions = get_conditions_by_group(args.group)
    
    print(f"Using conditions: {conditions}")
    dataset_dict = load_fold(fold, 
                            language=args.language, 
                            group=args.group, 
                            balance_level=args.balance_level)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, label_mapping = preprocess_dataset(
        dataset_dict["train"], 
        tokenizer, 
        args.max_length,
        conditions=conditions
    )

    eval_dataset, _ = preprocess_dataset(
        dataset_dict["test"], 
        tokenizer, 
        args.max_length,
        conditions=conditions
    )

    label_list = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    with open(os.path.join(fold_output_dir, 'label_mapping.txt'), 'w') as f:
        for label, idx in label_mapping.items():
            f.write(f"{label}: {idx}\n")
    
    num_labels = len(label_mapping)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(fold_output_dir, 'checkpoints'),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",
    )

    # Choose appropriate trainer based on arguments
    if args.weighted_training:
        # Calculate class weights based on training data distribution
        label_counts = {}
        for item in train_dataset:
            label = item["label"].item()
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Convert to numpy array with proper indices
        num_classes = len(label_mapping)
        class_counts = np.zeros(num_classes)
        for label, count in label_counts.items():
            class_counts[label] = count
        
        # Create a reverse mapping from index to class name for better reporting
        idx_to_label = {idx: label for label, idx in label_mapping.items()}
        
        # Print class distribution before weighting
        print("\nClass distribution in training data:")
        for label_idx in range(num_classes):
            class_name = idx_to_label[label_idx]
            count = class_counts[label_idx]
            print(f"  {class_name}: {int(count)} samples")
        
        # Check for classes with zero samples to avoid division by zero
        if 0 in class_counts:
            print("\nWarning: Some classes have zero samples. Using uniform weights for all classes.")
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            # Compute weights inversely proportional to class frequencies
            class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
            # Normalize weights so they sum to number of classes
            class_weights = class_weights * (num_classes / class_weights.sum())
        
        # Print the weights with corresponding class names
        print("\nClass weights for training:")
        for label_idx in range(num_classes):
            class_name = idx_to_label[label_idx]
            weight = class_weights[label_idx].item()
            print(f"  {class_name}: {weight:.4f}")
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            class_weights=class_weights
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    train_result = trainer.train()
    trainer.save_model(os.path.join(fold_output_dir, 'final_model'))
    eval_result = trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = predictions.label_ids

    plot_confusion_matrix(
        y_true, 
        y_pred, 
        labels=label_list,
        output_path=os.path.join(fold_output_dir, 'confusion_matrix.png')
    )
    
    with open(os.path.join(fold_output_dir, 'results.txt'), 'w') as f:
        f.write(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds\n")
        f.write(f"Evaluation results:\n")
        for key, value in eval_result.items():
            f.write(f"{key}: {value}\n")
    return {
        'fold': fold,
        'accuracy': eval_result['eval_accuracy'],
        'precision': eval_result['eval_precision'],
        'recall': eval_result['eval_recall'],
        'f1': eval_result['eval_f1'],
        'training_time': train_result.metrics['train_runtime']
    }

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(
        args.output_dir, 
        f"{args.model_name.split('/')[-1]}_{args.language}_{args.group}_{timestamp}"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    folds = [int(fold) for fold in args.folds.split(',')]
    results = []
    for fold in folds:
        fold_result = train_fold(fold, args)
        results.append(fold_result)
    results_df = pd.DataFrame(results)
    avg_metrics = {
        'accuracy': results_df['accuracy'].mean(),
        'precision': results_df['precision'].mean(),
        'recall': results_df['recall'].mean(),
        'f1': results_df['f1'].mean(),
        'training_time': results_df['training_time'].mean()
    }
    results_df.to_csv(os.path.join(args.output_dir, 'all_folds_results.csv'), index=False)
    print("\nAverage metrics across folds:")
    with open(os.path.join(args.output_dir, 'average_metrics.txt'), 'w') as f:
        for metric, value in avg_metrics.items():
            line = f"{metric}: {value:.4f}"
            print(line)
            f.write(f"{line}\n")
    print(f"\nTraining complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()