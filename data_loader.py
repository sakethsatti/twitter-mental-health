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
        "ANXIETY", "BIPOLAR", "DEPRESSION", "PTSD", "CONTR"
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
                    timelines_dir: str = TIMELINES_DIR,
                    normalize: bool = True) -> List[Dict]:
    """
    Load all tweets from a specific file and label them with the condition
    
    Args:
        filename: The CSV filename (e.g., usuario_12345.csv)
        language: Language code ("eng" or "esp")
        condition: Mental health condition to label the tweets with
        timelines_dir: Base directory for timeline files
        normalize: Whether to normalize tweets (default: True)
        
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
        
        # Create a list of tweet dictionaries
        tweets = []
        for _, row in df.iterrows():
            raw_tweet = row[tweet_column]
            if pd.notnull(raw_tweet):
                # Only normalize if requested
                tweet_text = normalizeTweet(str(raw_tweet)) if normalize else str(raw_tweet)
                tweets.append({
                    'tweet': tweet_text,
                    'class': condition,  
                    'language': language,
                    'user_filename': filename  # Add filename for grouping
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
             base_dir: str = BASE_DIR, balance_level: str = "none", 
             random_state: int = 626) -> DatasetDict:
    """
    Load data for a specific fold and language, filtered by mental health group
    
    Args:
        fold: Fold number (1-5)
        language: Language code ("eng" or "esp")
        group: Mental health group to include
        base_dir: Base directory for the dataset
        balance_level: Class balancing strategy:
            - "none": No balancing
            - "partial": Majority class reduced to size of largest minority class
            - "full": All classes reduced to size of smallest class
        random_state: Random seed for reproducibility in sampling
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    if fold < 1 or fold > 5:
        raise ValueError("Fold must be between 1 and 5")
    
    if language not in LANGUAGES:
        raise ValueError(f"Language must be one of {LANGUAGES}")
    
    if balance_level not in ["none", "partial", "full"]:
        raise ValueError("balance_level must be 'none', 'partial', or 'full'")
    
    # Get conditions for the specified group
    conditions = get_conditions_by_group(group)
    
    # Read partition data
    partition_df = read_partition_file(language, PARTITIONS_DIR)
    
    # Filter by the selected conditions
    partition_df = partition_df[partition_df['class'].isin(conditions)]
    
    # Get train and test data for the specified fold
    train_df = get_fold_data(partition_df, fold, "train")
    test_df = get_fold_data(partition_df, fold, "test")
    
    # Print class distribution for information
    class_user_counts = train_df['class'].value_counts().to_dict()
    print(f"\nOriginal class distribution in training set (users):")
    for cls, count in class_user_counts.items():
        print(f"  {cls}: {count} users")
    
    # Apply class balancing based on strategy
    if balance_level != "none" and len(conditions) > 1:
        # Import random here
        import random
        random.seed(random_state)
        
        # For both "partial" and "full" strategies
        if balance_level == "partial":
            # Find majority class and largest minority class
            majority_class = max(class_user_counts.items(), key=lambda x: x[1])[0]
            
            # Find the largest minority class
            minority_classes = {k: v for k, v in class_user_counts.items() if k != majority_class}
            largest_minority_class = max(minority_classes.items(), key=lambda x: x[1])
            largest_minority_name, largest_minority_count = largest_minority_class
            
            print(f"\nPartial balancing: Reducing {majority_class} to match {largest_minority_name} ({largest_minority_count} users)")
            
            # Create a balanced dataset by undersampling only the majority class
            majority_users = train_df[train_df['class'] == majority_class]
            non_majority_users = train_df[train_df['class'] != majority_class]
            
            if len(majority_users) > largest_minority_count:
                sampled_majority = majority_users.sample(n=largest_minority_count, random_state=random_state)
                train_df = pd.concat([non_majority_users, sampled_majority])
            
            # Same for test set
            class_user_counts = test_df['class'].value_counts().to_dict()
            if majority_class in class_user_counts:
                majority_users = test_df[test_df['class'] == majority_class]
                non_majority_users = test_df[test_df['class'] != majority_class]
                
                # Find the largest minority class in test set
                minority_classes = {k: v for k, v in class_user_counts.items() if k != majority_class}
                if minority_classes:
                    largest_minority_count = max(minority_classes.values())
                    
                    if len(majority_users) > largest_minority_count:
                        sampled_majority = majority_users.sample(n=largest_minority_count, random_state=random_state)
                        test_df = pd.concat([non_majority_users, sampled_majority])
        
        elif balance_level == "full":
            # Find the smallest class
            smallest_class_count = min(class_user_counts.values())
            smallest_class = min(class_user_counts.items(), key=lambda x: x[1])[0]
            print(f"\nFull balancing: Reducing all classes to match smallest class {smallest_class} ({smallest_class_count} users)")
            
            # Create a fully balanced dataset with equal users per class
            balanced_train_df = pd.DataFrame()
            
            for class_name in conditions:
                class_users = train_df[train_df['class'] == class_name]
                if len(class_users) > smallest_class_count:
                    sampled_users = class_users.sample(n=smallest_class_count, random_state=random_state)
                    balanced_train_df = pd.concat([balanced_train_df, sampled_users])
                else:
                    balanced_train_df = pd.concat([balanced_train_df, class_users])
            
            train_df = balanced_train_df
            
            # Same for test set
            class_user_counts = test_df['class'].value_counts().to_dict()
            smallest_class_count = min(class_user_counts.values())
            
            balanced_test_df = pd.DataFrame()
            for class_name in conditions:
                class_users = test_df[test_df['class'] == class_name]
                if len(class_users) > smallest_class_count:
                    sampled_users = class_users.sample(n=smallest_class_count, random_state=random_state)
                    balanced_test_df = pd.concat([balanced_test_df, sampled_users])
                else:
                    balanced_test_df = pd.concat([balanced_test_df, class_users])
            
            test_df = balanced_test_df
    
    # Print final class distribution
    class_user_counts = train_df['class'].value_counts().to_dict()
    print(f"\nFinal class distribution in training set (users):")
    for cls, count in class_user_counts.items():
        print(f"  {cls}: {count} users")
    
    # Process train data
    print("Loading train data...")
    train_data = []
    for _, row in train_df.iterrows():
        tweets = load_tweets_file(
            filename=row['filename'],
            language=language,
            condition=row['class'],
            timelines_dir=os.path.join(base_dir, "Timelines"),
            normalize=True
        )
        train_data.extend(tweets)
    
    # Process test data
    print("Loading test data...")
    test_data = []
    for _, row in test_df.iterrows():
        tweets = load_tweets_file(
            filename=row['filename'],
            language=language,
            condition=row['class'],
            timelines_dir=os.path.join(base_dir, "Timelines"),
            normalize=True
        )
        test_data.extend(tweets)
    
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