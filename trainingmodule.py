# chatbot_service.py
# Service-based chatbot using NLTK + scikit-learn
# Check Full Demo at - https://tripleminds.co/white-label/candy-ai-clone/
import random
import json
import nltk
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources (only first time)
nltk.download("punkt")
nltk.download("wordnet")

# -----------------------------
# 1. Define Intents Dataset
# -----------------------------
# You can expand this with your own service categories

intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello! How can I help you today?",
                          "Hi there! What service are you looking for?",
                          "Hey! Need any assistance?"]
        },
        {
            "tag": "services",
            "patterns": ["What services do you offer?", "Tell me about your services",
                         "I want to know what you provide", "Service list"],
            "responses": ["We provide Web Development, Mobile App Development, SEO, and AI Solutions.",
                          "Our services include Website & App Development, SEO, and Business Consulting.",
                          "We offer Digital Marketing, App Development, and AI-powered tools."]
        },
        {
            "tag": "pricing",
            "patterns": ["How much does it cost?", "Tell me about pricing",
                         "What are your charges?", "cost of your services"],
            "responses": ["Pricing depends on project scope. Small projects start from $500.",
                          "Our plans are flexible. Website projects start at $1000, apps at $5000.",
                          "Cost varies based on features. Let's discuss your requirements."]
        },
        {
            "tag": "support",
            "patterns": ["I need help", "Support", "Problem with my service",
                         "How can I contact support?", "customer service"],
            "responses": ["You can reach our support team at support@example.com",
                          "Please call +1-800-123-4567 for support.",
                          "Our help desk is available 24/7 at support@example.com."]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye", "Quit", "Exit"],
            "responses": ["Goodbye! Have a great day.",
                          "See you later. Thanks for visiting.",
                          "Bye! Feel free to ask anytime."]
        }
    ]
}

# -----------------------------
# 2. Preprocessing
# -----------------------------

# Collect patterns and tags
patterns = []
tags = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english")

X = vectorizer.fit_transform(patterns)
y = np.array(tags)

# -----------------------------
# 3. Train Classifier
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# 4. Response Function
# -----------------------------
def get_response(user_input):
    # Transform user input
    input_vec = vectorizer.transform([user_input])
    tag = model.predict(input_vec)[0]

    # Fetch a random response
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm not sure I understand. Can you rephrase?"

# -----------------------------
# 5. Chat Loop
# -----------------------------
print("🤖 Candy AI Clone (Service Chatbot) is ready! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot:", random.choice(["Goodbye!", "See you!", "Bye bye!"]))
        break
    response = get_response(user_input)
    print("Bot:", response)


🤖 Candy AI Clone (Service Chatbot) is ready! Type 'quit' to exit.
You: Hi
Bot: Hi there! What service are you looking for?

You: What services do you offer?
Bot: We provide Web Development, Mobile App Development, SEO, and AI Solutions.

You: How much does it cost?
Bot: Pricing depends on project scope. Small projects start from $500.

You: bye
Bot: Goodbye!
