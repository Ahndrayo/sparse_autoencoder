import re

def clean_text(text):
    # remove URLs
    text = re.sub(r"http\S+", "", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_leading_tickers(text):
    return re.sub(
        r'^\s*(?:\$[A-Z]{1,6}\s*)+(?:[-:]\s*)?',
        '',
        text
    )
