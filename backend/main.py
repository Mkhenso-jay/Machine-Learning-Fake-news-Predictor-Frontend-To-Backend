from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from preprocess import preprocess_text
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware



# Paths (relative to backend/)
MODEL_PATH = '../ML-Model/fake_news_detector.pkl'  
VECTORIZER_PATH = '../ML-Model/tfidf_vectorizer.pkl'



app = FastAPI(title="Fake News Predictor API", description="Backend for classifying news as real or fake.", version="1.0.0")

# allow only your local frontend in dev
origins = [
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",  # React default
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)


# Load model and vectorizer at startup
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int  # 0: Real, 1: Fake (matches training: 0=Real, 1=Fake)
    probabilities: Dict[str, float]  # e.g., {"real": 0.88, "fake": 0.12}
    confidence: float  # Max probability

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: PredictionRequest):
    """
    Predict if news text is fake or real.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    try:
        # Preprocess
        processed = preprocess_text(request.text)
        
        # Vectorize and predict
        vec_input = vectorizer.transform([processed])
        pred = model.predict(vec_input)[0]
        probs = model.predict_proba(vec_input)[0]
        
        # Format response (corrected labels: 0=real, 1=fake)
        label_to_key = {0: "real", 1: "fake"}
        confidence = max(probs)
        
        return PredictionResponse(
            prediction=pred,
            probabilities={label_to_key[i]: float(prob) for i, prob in enumerate(probs)},
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health endpoint."""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)