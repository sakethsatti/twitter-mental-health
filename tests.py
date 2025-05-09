import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from data_loader import load_fold, get_conditions_by_group
from tqdm import tqdm

def get_top_confident_correct_tweets(model_path, group, language="eng", fold_number=1, num_tweets_per_condition=200, max_length=96, batch_size=32):
    """
    Loads a trained model, runs predictions on the test set, and returns the
    top N correctly and confidently classified tweets per condition.

    Args:
        model_path (str): Path to the saved model directory.
        group (str): The condition group (e.g., "cognitive_attention").
        language (str, optional): Language of the dataset. Defaults to "eng".
        fold_number (int, optional): Fold number for the test set. Defaults to 1.
        num_tweets_per_condition (int, optional): Number of tweets to select per condition. Defaults to 200.
        max_length (int, optional): Max sequence length for tokenizer. Defaults to 96.
        batch_size (int, optional): Batch size for predictions. Defaults to 32.

    Returns:
        pandas.DataFrame: A DataFrame containing the selected tweets, their true labels,
                          predicted labels, and confidences, or None if an error occurs.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print(f"Loading test data for fold {fold_number}, group '{group}', language '{language}'...")
        dataset_dict = load_fold(fold_number, language=language, group=group, balance_level="none")
        if "test" not in dataset_dict:
            print(f"Error: 'test' set not found for fold {fold_number}, group '{group}'. Available keys: {dataset_dict.keys()}")
            return None
        test_dataset = dataset_dict["test"]

        conditions = get_conditions_by_group(group)
        if not conditions:
            print(f"Error: No conditions found for group '{group}'.")
            return None
            
        label_mapping = {condition: idx for idx, condition in enumerate(conditions)}
        id2label = {idx: condition for condition, idx in label_mapping.items()}
        
        num_labels = len(conditions)
        if model.config.num_labels != num_labels:
            print(f"Warning: Model was trained with {model.config.num_labels} labels, but group '{group}' has {num_labels} conditions.")
            # Potentially problematic, but proceed with caution. User might be aware.

        print(f"Found {len(test_dataset)} tweets in the test set.")
        print(f"Conditions: {conditions}")
        print(f"Label mapping: {label_mapping}")

        all_tweets_text = []
        all_true_label_ids = []
        all_predicted_label_ids = []
        all_confidences = []

        print(f"Running predictions (batch size: {batch_size})...")
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch_tweets = test_dataset[i:i+batch_size]["tweet"]
            batch_labels_text = test_dataset[i:i+batch_size]["class"]
            
            batch_true_label_ids = [label_mapping.get(label) for label in batch_labels_text]

            inputs = tokenizer(batch_tweets, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                confidences, predicted_ids = torch.max(probabilities, dim=-1)

            all_tweets_text.extend(batch_tweets)
            all_true_label_ids.extend(batch_true_label_ids)
            all_predicted_label_ids.extend(predicted_ids.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
        
        results_df = pd.DataFrame({
            "tweet": all_tweets_text,
            "true_label_id": all_true_label_ids,
            "predicted_label_id": all_predicted_label_ids,
            "confidence": all_confidences
        })

        # Convert label IDs back to string names
        results_df["true_label"] = results_df["true_label_id"].apply(lambda x: id2label.get(x))
        results_df["predicted_label"] = results_df["predicted_label_id"].apply(lambda x: id2label.get(x))
        
        # Filter for correct predictions
        correct_predictions_df = results_df[results_df["true_label_id"] == results_df["predicted_label_id"]].copy()
        print(f"Number of correctly predicted tweets: {len(correct_predictions_df)}")

        if correct_predictions_df.empty:
            print("No tweets were correctly predicted.")
            return pd.DataFrame() # Return empty DataFrame

        # Sort by confidence within each true label and get top N
        top_tweets_list = []
        for condition_name, condition_id in label_mapping.items():
            condition_df = correct_predictions_df[correct_predictions_df["true_label_id"] == condition_id]
            top_n = condition_df.sort_values(by="confidence", ascending=False).head(num_tweets_per_condition)
            top_tweets_list.append(top_n)
            print(f"Selected {len(top_n)} tweets for condition '{condition_name}' (target: {num_tweets_per_condition})")

        final_df = pd.concat(top_tweets_list)
        print(f"Total selected tweets: {len(final_df)}")
        
        return final_df

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    MODEL_PATH = "results/bertweet-base_eng_cognitive_attention_20250421_023107/fold_1/final_model"
    GROUP = "cognitive_attention" 
    LANGUAGE = "eng"
    FOLD = 1
    NUM_TWEETS = 200 # 
    OUTPUT_FILE = f"top_{NUM_TWEETS}_confident_correct_tweets_{GROUP}_{LANGUAGE}.csv"

    print(f"Starting analysis for group: {GROUP}")
    selected_tweets_df = get_top_confident_correct_tweets(
        model_path=MODEL_PATH,
        group=GROUP,
        language=LANGUAGE,
        fold_number=FOLD,
        num_tweets_per_condition=NUM_TWEETS
    )

    if selected_tweets_df is not None and not selected_tweets_df.empty:
        print(f"\n--- Top {NUM_TWEETS} Confident and Correct Tweets per Condition for '{GROUP}' ---")
        for condition in get_conditions_by_group(GROUP):
            condition_data = selected_tweets_df[selected_tweets_df["true_label"] == condition]
            print(f"\nCondition: {condition} (Found: {len(condition_data)})")
            print(condition_data[['tweet', 'confidence', 'true_label', 'predicted_label']].head()) 
        
        # Save to CSV
        try:
            selected_tweets_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\nResults saved to {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            
    elif selected_tweets_df is not None and selected_tweets_df.empty:
        print("No tweets met the criteria.")
    else:
        print("Failed to retrieve tweets.")
