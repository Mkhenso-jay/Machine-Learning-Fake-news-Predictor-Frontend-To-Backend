import streamlit as st
import requests
import pandas as pd
import json

# Backend API config
API_URL = "http://localhost:8000/predict"  # Matches your FastAPI port

# App title and description
st.title("📰 Fake News Detector")
st.markdown("""
Paste news text below—our backend will classify it as **Real** or **Fake** (powered by your 99% accurate ML model!).
""")

# Sidebar for info and samples
with st.sidebar:
    st.header("Quick Info")
    st.write("**Backend API:** localhost:8000")
    st.write("**Endpoint:** POST /predict")
    
    st.header("Load Sample")
    if st.button("Grab Sample from Dataset"):
        try:
            # Adjust path to your actual dataset (e.g., full path if needed)
            df = pd.read_csv('../data/new_dataset.csv')  # Or use absolute: r'D:\Fake_News_Predictor\data\new_dataset.csv'
            sample = df['text'].sample(1).iloc[0]  # Assuming 'text' or 'content' column—change if needed
            st.text_area("Sample Loaded:", value=sample, height=100, key="sample")
        except FileNotFoundError:
            st.warning("Dataset not found at '../data/new_dataset.csv'. Load manually or fix path.")
        except Exception as e:
            st.error(f"Sample load failed: {e}")

# Main input (use session state to persist if re-run)
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area(
    "Enter news text:",
    value=st.session_state.user_input,
    height=200,
    placeholder="e.g., 'Shocking: Flat Earth proven by new NASA leak...'"
)
st.session_state.user_input = user_input  # Save to session

# Predict button
if st.button("🔍 Detect Fake News", type="primary"):
    if user_input.strip():
        with st.spinner("Sending to backend..."):
            try:
                # API request (matches backend: {"text": input})
                payload = {"text": user_input}
                response = requests.post(API_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # FIXED: Match training labels (0=Real, 1=Fake)
                    pred_label = "🟢 Real News" if result["prediction"] == 0 else "🔴 Fake News"
                    conf = result["confidence"] * 100
                    
                    st.success(pred_label)
                    st.write(f"**Confidence:** {conf:.1f}%")
                    
                    # Metrics (matches backend keys: 'real' and 'fake')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Real Prob", f"{result['probabilities']['real']*100:.1f}%")
                    with col2:
                        st.metric("Fake Prob", f"{result['probabilities']['fake']*100:.1f}%")
                    
                    # Tip
                    st.info("💡 Low confidence? Add more article context for better accuracy.")
                    
                    # Optional: Show raw response for debugging
                    with st.expander("Raw API Response"):
                        st.json(result)
                        
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Backend connection failed: {str(e)}. Start FastAPI with 'uvicorn main:app --reload' on port 8000?")
            except json.JSONDecodeError:
                st.error("Invalid JSON from backend. Check preprocess.py or model load.")
    else:
        st.warning("Enter some text first!")

# Footer
st.markdown("---")
st.markdown("Frontend by Streamlit | Backend: FastAPI + Your LogisticRegression Model")