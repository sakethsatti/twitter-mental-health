import os
import pandas as pd
from datasets import Dataset, DatasetDict
from TweetNormalizer import normalizeTweet
from typing import List, Dict, Union, Tuple, Optional

# Constants
BASE_DIR = "./tweets"
PARTITIONS_DIR = os.path.join(BASE_DIR, "Partitions")
TIMELINES_DIR = os.path.join(BASE_DIR, "Timelines")

# Languages available
LANGUAGES = ["eng", "esp"]

# Mental health condition groups
GROUPS = {
    "all": [
        "ADHD", "ANXIETY", "ASD", "BIPOLAR", "CONTROL", 
        "DEPRESSION", "EATING", "OCD", "PTSD", "SCHIZOPHRENIA"
    ],
    "internalizing": [
        "ANXIETY", "BIPOLAR", "DEPRESSION", "PTSD", "CONTROL"
    ],
    "cognitive_attention": [
        "ADHD", "ASD", "CONTROL"
    ],
    "eating_disorders": [
        "EATING", "CONTROL"
    ],
    "anxiety_disorders": [
        "ANXIETY", "OCD", "PTSD", "CONTROL"
    ],
    "psychotic": [
        "SCHIZOPHRENIA", "BIPOLAR", "CONTROL"
    ]
}

# Default to all conditions
CONDITIONS = GROUPS["all"]

def get_conditions_by_group(group="all"):
    """
    Get the list of conditions for a specific mental health group
    
    Args:
        group: The mental health group to filter by
    
    Returns:
        List of conditions in that group
    """
    if group not in GROUPS:
        raise ValueError(f"Group must be one of {list(GROUPS.keys())}")
    
    return GROUPS[group]

def read_partition_file(language: str, partitions_dir: str = PARTITIONS_DIR) -> pd.DataFrame:
    """Read the partition file for the specified language"""
    file_path = os.path.join(partitions_dir, f"data_split_5FCV_{language}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Partition file not found: {file_path}")
    return pd.read_csv(file_path)

def get_fold_data(df: pd.DataFrame, fold: int, split: str) -> pd.DataFrame:
    """Extract data for specific fold and split (train or test)"""
    # The partition column has values like 'train_fold1' or 'test_fold1'
    partition_value = f"{split}_fold{fold}"
    return df[df['partition'] == partition_value].copy()

def load_tweets_file(filename: str, language: str, condition: str, 
                    timelines_dir: str = TIMELINES_DIR) -> List[Dict]:
    """
    Load all tweets from a specific file and label them with the condition
    
    Args:
        filename: The CSV filename (e.g., usuario_12345.csv)
        language: Language code ("eng" or "esp")
        condition: Mental health condition to label the tweets with
        timelines_dir: Base directory for timeline files
        
    Returns:
        List of dictionaries with tweet data
    """
    lang_dir = "English" if language == "eng" else "Spanish"
    
    # Fix capitalization: first letter uppercase, rest lowercase
    # e.g., "ADHD" becomes "Adhd", "ANXIETY" becomes "Anxiety"
    condition_dir = condition.capitalize() if len(condition) <= 3 else condition[0].upper() + condition[1:].lower()
    
    file_path = os.path.join(
        timelines_dir, 
        lang_dir,
        f"{condition_dir}_{language}", 
        filename
    )
    
    if not os.path.exists(file_path):
        print(f"Warning: Tweet file not found: {file_path}")
        return []
    
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
        
        # Identify the tweet text column
        tweet_column = 'tweet'
        
        # Create a list of tweet dictionaries, normalizing each tweet
        tweets = []
        for _, row in df.iterrows():
            raw_tweet = row[tweet_column]
            cleaned_tweet = normalizeTweet(str(raw_tweet)) if pd.notnull(raw_tweet) else ""
            tweets.append({
                'tweet': cleaned_tweet,
                'class': condition,  
                'language': language
            })
        
        return tweets
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def undersample_majority_class(data: List[Dict], majority_class: str = "CONTROL", random_state: int = 626) -> List[Dict]:
    """
    Undersample the majority class (CONTROL) to match the largest minority class
    
    Args:
        data: List of data points
        majority_class: The class to undersample (default: "CONTROL")
        random_state: Random seed for reproducibility (default: 626)
        
    Returns:
        List with balanced data
    """
    # Count samples per class
    class_counts = {}
    for item in data:
        class_label = item['class']
        if class_label not in class_counts:
            class_counts[class_label] = 0
        class_counts[class_label] += 1
    
    # If CONTROL is not in the data, return the original data
    if majority_class not in class_counts:
        return data
    
    # Find the largest minority class
    sorted_counts = sorted(
        [(k, v) for k, v in class_counts.items() if k != majority_class],
        key=lambda x: x[1], 
        reverse=True
    )
    
    # If no minority classes, return original data
    if not sorted_counts:
        return data
    
    largest_minority_class, largest_minority_count = sorted_counts[0]
    
    print(f"Undersampling {majority_class} ({class_counts[majority_class]} samples) "
          f"to match {largest_minority_class} ({largest_minority_count} samples)")
    
    # Separate majority and minority instances
    majority_samples = [item for item in data if item['class'] == majority_class]
    minority_samples = [item for item in data if item['class'] != majority_class]
    
    # Randomly sample from majority class
    import random
    random.seed(random_state)  # For reproducibility
    if len(majority_samples) > largest_minority_count:
        majority_samples = random.sample(majority_samples, largest_minority_count)
    
    # Combine and return
    return majority_samples + minority_samples

def load_fold(fold: int, language: str = "eng", group: str = "all",
             base_dir: str = BASE_DIR, undersample_control: bool = True, 
             random_state: int = 626) -> DatasetDict:
    """
    Load data for a specific fold and language, filtered by mental health group
    
    Args:
        fold: Fold number (1-5)
        language: Language code ("eng" or "esp")
        group: Mental health group to include
        base_dir: Base directory for the dataset
        undersample_control: Whether to undersample the CONTROL class
        random_state: Random seed for reproducibility in undersampling
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    if fold < 1 or fold > 5:
        raise ValueError("Fold must be between 1 and 5")
    
    if language not in LANGUAGES:
        raise ValueError(f"Language must be one of {LANGUAGES}")
    
    # Get conditions for the specified group
    conditions = get_conditions_by_group(group)
    
    # Read partition data
    partition_df = read_partition_file(language, PARTITIONS_DIR)
    
    # Filter by the selected conditions
    partition_df = partition_df[partition_df['class'].isin(conditions)]
    
    # Get train and test data for the specified fold
    train_df = get_fold_data(partition_df, fold, "train")
    test_df = get_fold_data(partition_df, fold, "test")
    
    # Process train data
    train_data = []
    for _, row in train_df.iterrows():
        tweets = load_tweets_file(
            filename=row['filename'],
            language=language,
            condition=row['class'],
            timelines_dir=os.path.join(base_dir, "Timelines")
        )
        train_data.extend(tweets)
    
    # Process test data
    test_data = []
    for _, row in test_df.iterrows():
        tweets = load_tweets_file(
            filename=row['filename'],
            language=language,
            condition=row['class'],
            timelines_dir=os.path.join(base_dir, "Timelines")
        )
        test_data.extend(tweets)
    
    # Undersample control class if requested and if we're not using all classes
    if undersample_control and group != "all":
        print(f"\nUndersampling CONTROL class for training set:")
        train_data = undersample_majority_class(train_data, random_state=random_state)
        
        print(f"\nUndersampling CONTROL class for testing set:")
        test_data = undersample_majority_class(test_data, random_state=random_state)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    return DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

def load_all_folds(language: str = "eng", group: str = "all", 
                  base_dir: str = BASE_DIR, undersample_control: bool = True,
                  random_state: int = 626) -> Dict[int, DatasetDict]:
    """
    Load data for all 5 folds for a specific language and mental health group
    
    Args:
        language: Language code ("eng" or "esp")
        group: Mental health group to include
        base_dir: Base directory for the dataset
        undersample_control: Whether to undersample the CONTROL class
        random_state: Random seed for reproducibility in undersampling
        
    Returns:
        Dictionary mapping fold numbers to DatasetDict objects
    """
    return {fold: load_fold(fold, language, group, base_dir, undersample_control, random_state) 
            for fold in range(1, 6)}