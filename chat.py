"""
chat.py
Interactive CLI chatbot tester for Candy AI Clone.

- Loads model + vectorizer + intents
- Lets developer chat in terminal
- Useful for quick smoke tests before hooking into API (app.py)
"""

import joblib
import json
import random
import os
from config import MODEL_PATH, VECTORIZER_PATH, INTENTS_PATH

# -------------------------
# Load artifacts
# -------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError("Model/vectorizer not found. Run train.py first.")

print("[INFO] Loading model + vectorizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

print("[INFO] Intents loaded.")

# -------------------------
# Helpers
# -------------------------
def get_response(user_input: str):
    X = vectorizer.transform([user_input])
    tag = model.predict(X)[0]

    # pick response
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"]), tag
    return "I'm not sure I understand. Could you rephrase?", "fallback"

# -------------------------
# CLI Loop
# -------------------------
print("\n🤖 Candy AI Clone — Terminal Chatbot Tester")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye! 👋")
        break

    reply, tag = get_response(user_input)
    print(f"Bot ({tag}): {reply}")
