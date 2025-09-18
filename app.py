"""
app.py
FastAPI server for NSFW (adult) service chatbot core.
- Loads model + vectorizer + intents
- Exposes /chat endpoint
- Includes simple safety/mode checks + logging hook
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import joblib
import json
import os
import time

# -------------------------
# Config & Paths
# -------------------------
from config import MODEL_PATH, VECTORIZER_PATH, INTENTS_PATH, APP_NAME, ALLOWED_ORIGINS

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Models & Data Loading
# -------------------------
MODEL = None
VECTORIZER = None
INTENTS: Dict[str, Any] = {}

def load_artifacts():
    global MODEL, VECTORIZER, INTENTS
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError("Model or vectorizer not found. Train first (see train.py).")
    MODEL = joblib.load(MODEL_PATH)
    VECTORIZER = joblib.load(VECTORIZER_PATH)
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        INTENTS = json.load(f)

load_artifacts()

# -------------------------
# Schemas
# -------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input text")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    mode: Optional[str] = Field(default="default", description="bot mode, e.g., 'default', 'nsfw', 'safe'")

class ChatResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[int] = None

# -------------------------
# Helpers
# -------------------------
SAFEPLACEHOLDER = (
    "I can’t continue with that request. "
    "Please keep it respectful and within acceptable guidelines."
)

def simple_moderation(text: str, mode: str = "default") -> bool:
    """
    Return True if the request is ALLOWED to proceed.
    This is a placeholder; replace with your policy rules or an external moderator.
    """
    # Example toggles:
    if mode == "safe":
        # Block sensitive/explicit words (very basic placeholder).
        blocked = ["minors", "illegal", "exploit", "rape"]  # extend appropriately
        if any(b in text.lower() for b in blocked):
            return False
    # In 'nsfw' mode you still should enforce your platform’s policy.
    return True

def classify_and_respond(message: str) -> Dict[str, Any]:
    """
    Vectorize -> predict intent -> choose response.
    INTENTS format expected from intents.json:
    {
      "intents": [{"tag": "...", "responses": ["...","..."]}, ...]
    }
    """
    X = VECTORIZER.transform([message])
    proba = getattr(MODEL, "predict_proba", None)
    tag = MODEL.predict(X)[0]
    conf = float(max(proba(X)[0])) if callable(proba) else None

    # pick response by tag
    responses = next((i["responses"] for i in INTENTS["intents"] if i["tag"] == tag), None)
    reply = (responses or ["I'm not sure I understood that. Could you rephrase?"])[0]
    return {"reply": reply, "intent": tag, "confidence": conf}

def log_event(user_id: Optional[str], message: str, intent: Optional[str], reply: str, ok: bool):
    """Hook for logging/analytics; replace with DB or queue as needed."""
    print({
        "ts": int(time.time() * 1000),
        "user_id": user_id,
        "message": message,
        "intent": intent,
        "reply_len": len(reply),
        "allowed": ok,
    })

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.perf_counter()

    if not simple_moderation(req.message, req.mode):
        log_event(req.user_id, req.message, None, SAFEPLACEHOLDER, ok=False)
        return ChatResponse(reply=SAFEPLACEHOLDER, latency_ms=int((time.perf_counter()-start)*1000))

    try:
        result = classify_and_respond(req.message)
        latency = int((time.perf_counter()-start)*1000)
        log_event(req.user_id, req.message, result.get("intent"), result["reply"], ok=True)
        return ChatResponse(
            reply=result["reply"],
            intent=result.get("intent"),
            confidence=result.get("confidence"),
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat error: {e}")

# Optional reload endpoint after retraining
@app.post("/reload")
def reload_artifacts():
    try:
        load_artifacts()
        return {"reloaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reload error: {e}")

# -------------------------
# Dev server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
