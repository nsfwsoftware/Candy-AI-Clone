"""
train.py
Training pipeline for Candy AI Clone (NSFW chatbot).
- Reads intents.json
- Builds dataset (patterns → tags)
- Trains a classifier with TF-IDF
- Saves model + vectorizer artifacts
- Prints evaluation metrics
"""

import json
import random
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    INTENTS_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    settings
)

# -------------------------
# 1. Load Data
# -------------------------
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

patterns = []
tags = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(tag)

print(f"[INFO] Loaded {len(patterns)} patterns across {len(set(tags))} intent tags.")

# -------------------------
# 2. Split Train/Test
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    patterns,
    tags,
    test_size=settings.TEST_SIZE,
    random_state=settings.RANDOM_STATE,
    stratify=tags
)

# -------------------------
# 3. Build Pipeline
# -------------------------
if settings.CLASSIFIER == "logreg":
    classifier = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto")
elif settings.CLASSIFIER == "linear_svc":
    from sklearn.svm import LinearSVC
    classifier = LinearSVC()
elif settings.CLASSIFIER == "sgd":
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(loss="log_loss")
else:
    raise ValueError(f"Unsupported classifier: {settings.CLASSIFIER}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=settings.MAX_FEATURES,
        ngram_range=settings.NGRAM_RANGE
    )),
    ("clf", classifier)
])

# -------------------------
# 4. Train Model
# -------------------------
print("[INFO] Training model...")
pipeline.fit(X_train, y_train)
print("[INFO] Training complete.")

# -------------------------
# 5. Evaluate
# -------------------------
y_pred = pipeline.predict(X_test)
print("\n[REPORT] Classification Report")
print(classification_report(y_test, y_pred))

print("\n[REPORT] Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -------------------------
# 6. Save Artifacts
# -------------------------
# Save whole pipeline (vectorizer + classifier)
joblib.dump(pipeline, MODEL_PATH)
print(f"[INFO] Model saved → {MODEL_PATH}")

# If you want separate vectorizer and classifier:
joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
print(f"[INFO] Vectorizer saved → {VECTORIZER_PATH}")

print("\n✅ Training finished successfully.")
