import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import pandas as pd
from collections import Counter
from tqdm import tqdm
from data_loader import get_conditions_by_group 
import json 
import os

def analyze_token_influence(
    csv_file_path,
    model_path,
    group_name,
    num_top_tokens=20, # This will now be used for printing, but all data is saved
    tokenizer_name="vinai/bertweet-base",
    output_data_file="token_attributions.json" # New parameter for output file
):
    """
    Analyzes token influence on model predictions for different mental health conditions.

    Args:
        csv_file_path (str): Path to the CSV file containing top tweets.
                             Expected columns: 'tweet', 'true_label'.
        model_path (str): Path to the saved Hugging Face model directory.
        group_name (str): The condition group (e.g., "cognitive_attention") to get label mappings.
        num_top_tokens (int, optional): Number of top influential tokens to display per condition. Defaults to 20.
        tokenizer_name (str, optional): Name or path of the tokenizer. Defaults to "vinai/bertweet-base".
    """
    try:
        print(f"Loading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        if not all(col in df.columns for col in ['tweet', 'true_label']):
            print(f"Error: CSV file must contain 'tweet' and 'true_label' columns.")
            return
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loading model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    conditions = get_conditions_by_group(group_name)
    if not conditions:
        print(f"Error: No conditions found for group '{group_name}' using get_conditions_by_group.")
        return
    
    label_mapping = {condition: idx for idx, condition in enumerate(conditions)}
    # id2label = {idx: condition for condition, idx in label_mapping.items()} # Not strictly needed for explainer if using class_name

    print(f"Initializing explainer for {len(conditions)} conditions: {conditions}")
    # The explainer can often infer class names if the model config has id2label,
    # but explicitly providing it or using label_id is safer.
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)

    aggregated_attributions = {}

    # Ensure 'true_label' contains valid conditions for the group
    df = df[df['true_label'].isin(conditions)]
    if df.empty:
        print(f"No tweets found in the CSV matching the conditions for group '{group_name}'.")
        return

    unique_conditions_in_df = df['true_label'].unique()
    print(f"Found {len(unique_conditions_in_df)} conditions in the CSV to analyze: {unique_conditions_in_df}")

    for condition_name in unique_conditions_in_df:
        print(f"\nProcessing condition: {condition_name}")
        condition_tweets = df[df['true_label'] == condition_name]['tweet'].tolist()
        
        if not condition_tweets:
            print(f"No tweets found for condition: {condition_name}")
            continue

        condition_token_scores = Counter()
        
        target_id = label_mapping.get(condition_name)
        if target_id is None:
            print(f"Warning: Could not find label ID for condition '{condition_name}'. Skipping.")
            continue

        for tweet_text in tqdm(condition_tweets, desc=f"Explaining tweets for {condition_name}"):
            try:
                word_attributions = cls_explainer(tweet_text, class_index=target_id) 
                for token, score in word_attributions:
                    condition_token_scores[token] += score
            except Exception as e:
                print(f"Error explaining tweet: '{tweet_text[:50]}...'. Error: {e}")
                continue
        
        aggregated_attributions[condition_name] = condition_token_scores

    print("\n--- Top Influential Tokens per Condition (Console Display) ---")
    for condition_name, token_scores in aggregated_attributions.items():
        print(f"\nCondition: {condition_name}")
        if not token_scores:
            print("  No attributions calculated.")
            continue
        # Sort tokens by score for display, ensuring it's a list of tuples
        # Counter.most_common() already returns a list of (element, count) tuples
        top_tokens_display = token_scores.most_common(num_top_tokens)
        for token, score in top_tokens_display:
            print(f"  - \"{token}\": {score:.4f}")

    # Save the aggregated attributions to a JSON file
    # Convert Counter objects to dicts for JSON serialization
    attributions_to_save = {
        cond: dict(scores) for cond, scores in aggregated_attributions.items()
    }
    try:
        output_dir = os.path.dirname(output_data_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        with open(output_data_file, 'w') as f:
            json.dump(attributions_to_save, f, indent=4)
        print(f"\nAggregated token attributions saved to: {output_data_file}")
    except Exception as e:
        print(f"Error saving attributions to JSON: {e}")

if __name__ == "__main__":
    CSV_INPUT_FILE = "top_200_confident_correct_tweets_cognitive_attention.csv" 
    MODEL_PATH = "results/bertweet-base_eng_cognitive_attention_20250421_023107/fold_1/final_model"
    GROUP_NAME = "cognitive_attention" 
    NUM_TOP_TOKENS_TO_SHOW = 25 # For console output
    OUTPUT_DATA_FILENAME = f"results/token_attributions_{GROUP_NAME}.json" # Define output file name

    print(f"Starting token influence analysis for group: {GROUP_NAME}")
    print(f"Using input CSV: {CSV_INPUT_FILE}")
    print(f"Using model: {MODEL_PATH}")

    analyze_token_influence(
        csv_file_path=CSV_INPUT_FILE,
        model_path=MODEL_PATH,
        group_name=GROUP_NAME,
        num_top_tokens=NUM_TOP_TOKENS_TO_SHOW,
        output_data_file=OUTPUT_DATA_FILENAME 
    )

    print("\nAnalysis complete.")