"""
utils.py
Helper functions for Candy AI Clone chatbot (Triple Minds).
- Text preprocessing (clean, normalize, tokenize)
- Confidence checks & fallbacks
- ID generators for users/sessions
"""

import re
import string
import random
import hashlib
import uuid
from typing import List, Optional

import nltk

# Make sure tokenizers are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# -------------------------
# Text Cleaning
# -------------------------
def clean_text(text: str) -> str:
    """
    Normalize input text:
    - Lowercase
    - Remove punctuation
    - Collapse extra spaces
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words using nltk.
    """
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)


# -------------------------
# Confidence Handling
# -------------------------
def is_confident(confidence: Optional[float], threshold: float = 0.55) -> bool:
    """
    Returns True if model confidence is above threshold.
    """
    if confidence is None:
        return False
    return confidence >= threshold


def fallback_response() -> str:
    """
    Default fallback reply if model uncertain.
    """
    return "I’m not sure I understood that. Could you rephrase?"


# -------------------------
# ID Generators
# -------------------------
def generate_user_id(prefix: str = "user") -> str:
    """
    Generate a short user ID (e.g., user_a1b2c3).
    """
    rand = uuid.uuid4().hex[:6]
    return f"{prefix}_{rand}"


def hash_user_identifier(identifier: str) -> str:
    """
    Hash an email/IP/etc. to anonymize user reference.
    """
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]


# -------------------------
# Developer Debugging
# -------------------------
def debug_print(title: str, obj) -> None:
    """
    Print debug info in a consistent way.
    """
    print(f"\n[DEBUG] {title}:\n{obj}\n")


if __name__ == "__main__":
    # Quick self-test
    s = "Hello!!! How much for your SERVICES???"
    print("Cleaned:", clean_text(s))
    print("Tokens:", tokenize(s))
    print("User ID:", generate_user_id())
    print("Hash(email):", hash_user_identifier("test@example.com"))
    print("Confident 0.7?:", is_confident(0.7))
