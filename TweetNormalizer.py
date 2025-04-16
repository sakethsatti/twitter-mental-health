"""
ALL CODE WAS TAKEN FROM https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
MODIFICATIONS MADE FOR EFFICIENCY
"""

from emoji import demojize
from nltk.tokenize import TweetTokenizer
import re

tokenizer = TweetTokenizer()

# Pre-compile regex patterns for efficiency
REPLACEMENTS = [
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bn['’]t\b"), " n't"),
    (re.compile(r"\bn\s*['’]t\b"), " n't"),
    (re.compile(r"\bca n't\b"), "can't"),
    (re.compile(r"\bai n't\b"), "ain't"),
    (re.compile(r"'m\b"), " 'm"),
    (re.compile(r"'re\b"), " 're"),
    (re.compile(r"'s\b"), " 's"),
    (re.compile(r"'ll\b"), " 'll"),
    (re.compile(r"'d\b"), " 'd"),
    (re.compile(r"'ve\b"), " 've"),
    (re.compile(r"p \. m \."), "p.m."),
    (re.compile(r"p \. m\b"), "p.m"),
    (re.compile(r"a \. m \."), "a.m."),
    (re.compile(r"a \. m\b"), "a.m"),
]

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    elif token == "’":
        return "'"
    elif token == "…":
        return "..."
    else:
        return token

def normalizeTweet(tweet):
    # Pre-replace special unicode chars
    tweet = tweet.replace("’", "'").replace("…", "...")
    tokens = tokenizer.tokenize(tweet)
    normTweet = " ".join(normalizeToken(token) for token in tokens)

    # Apply all regex replacements in one pass
    for pattern, repl in REPLACEMENTS:
        normTweet = pattern.sub(repl, normTweet)

    # Remove redundant spaces
    return " ".join(normTweet.split())