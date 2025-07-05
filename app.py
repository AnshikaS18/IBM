import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

@st.cache_resource
def load_generator():
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=256
    )

generator = load_generator()

st.set_page_config(page_title="JobFit AI", layout="centered")
st.title("📄 JobFit AI – Resume vs JD Matching")
st.write("Paste your resume and job description. We'll check how well they match, give suggestions, and generate a cover letter using AI.")

resume_text = st.text_area("✍️ Paste Your Resume Here", height=200)
jd_text = st.text_area("🧾 Paste Job Description Here", height=200)

if st.button("🔍 Analyze"):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please enter both resume and job description.")
    else:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([resume_text, jd_text])

        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        sim_pct = round(sim * 100, 2)

        model = LinearRegression()
        model.fit(np.array([[0.1],[0.3],[0.6],[0.8],[0.9]]), np.array([25, 45, 70, 85, 95]))
        fit_score = round(model.predict([[sim]])[0], 2)

        resume_words = set(resume_text.lower().split())
        jd_words = set(jd_text.lower().split())
        missing = jd_words - resume_words
        missing_keywords = ", ".join(list(missing)[:10]) if missing else "✅ All important keywords are present!"

        feedback_prompt = f"""
        You are an HR coach.
        Cosine similarity = {sim_pct}%. Fit score = {fit_score}.
        Missing keywords = {missing_keywords}.
        Give resume improvement suggestions in 80 words.
        """
        feedback = generator(feedback_prompt)[0]['generated_text']

        cover_prompt = f"""
        Write a 120-word professional cover letter based on this resume:\n{resume_text}\nand this job description:\n{jd_text}.
        Focus on skills match and enthusiasm.
        """
        cover = generator(cover_prompt)[0]['generated_text']

        st.subheader("📊 Similarity Score")
        st.info(f"Cosine Similarity: **{sim_pct}%**")
        st.success(f"Predicted Fit Score: **{fit_score}/100**")

        st.subheader("🧠 AI Feedback")
        st.write(feedback)

        st.subheader("🔍 Missing Keywords")
        st.write(missing_keywords)

        st.subheader("📩 AI-Generated Cover Letter")
        st.write(cover)
