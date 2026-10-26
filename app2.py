import streamlit as st
import re
import joblib
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# ==========================
# SAFE MODEL LOADING
# ==========================
try:
    model = joblib.load("resume_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("label_encoder.pkl")
except:
    st.error("❌ Model files not found! Please check .pkl files")
    st.stop()

# ==========================
# NLTK DOWNLOAD (SAFE)
# ==========================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==========================
# CLEAN TEXT
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ==========================
# CUSTOM UI (FIXED)
# ==========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f172a, #1e3c72);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 20px;
}

.result-box {
    background-color: #22c55e;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 22px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("<div class='title'>📄 AI Resume Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Paste resume and get instant job category prediction</div>", unsafe_allow_html=True)

st.write("---")

# ==========================
# INPUT
# ==========================
resume = st.text_area("✍ Paste Resume Here", height=250)

# ==========================
# BUTTON
# ==========================
if st.button("🚀 Predict Category"):

    if resume.strip() == "":
        st.warning("Please enter resume text")
    else:
        cleaned = clean_text(resume)
        vector = tfidf.transform([cleaned])
        pred = model.predict(vector)
        result = le.inverse_transform(pred)[0]

        st.markdown(f"""
        <div class="result-box">
        🎯 Predicted Category: {result}
        </div>
        """, unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("📊 Project Info")
st.sidebar.info("""
✔ AI Resume Classifier  
✔ ML: LinearSVC  
✔ NLP: TF-IDF + NLTK  
✔ Output: Job Category  
""")