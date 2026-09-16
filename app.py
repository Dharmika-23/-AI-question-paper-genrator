import streamlit as st
from utils import extract_text, split_sentences, top_k_relevant_sentences
from qg import generate_questions_from_sentences

st.set_page_config(page_title="AI Questions  Generator", layout="wide")

st.title("🧠 AI Questions  Generator")
st.write("Upload your study material and generate questions from it using AI!")

# --- File upload ---
uploaded_file = st.file_uploader("📂 Upload File (PDF or TXT)", type=["pdf", "txt"])

# --- Topic input ---
topic = st.text_input("🎯 Enter topic name", placeholder="e.g., CPU scheduling")

# --- Sliders ---
num_sentences = st.slider("Number of relevant sentences", 1, 10, 5)
questions_per_sentence = st.slider("Questions per sentence", 1, 3, 2)

# --- Button ---
if st.button("🚀 Generate Questions"):
    if uploaded_file and topic:
        with st.spinner("Processing your document..."):
            text = extract_text(uploaded_file)
            sentences = split_sentences(text)
            top_sentences = top_k_relevant_sentences(sentences, topic, num_sentences)
            qa_pairs = generate_questions_from_sentences(top_sentences, questions_per_sentence)

        # --- Display Results ---
        if qa_pairs:
            st.success("✅ Questions generated successfully!")
            st.subheader("Generated Q&A")

            for i, qa in enumerate(qa_pairs, 1):
                # Each qa is a tuple (context, question)
                context, question = qa
                st.markdown(f"**{i}) Context:** {context}")
                st.markdown(f"➡️ **Question:** {question}")
                st.markdown("---")
        else:
            st.warning("⚠️ No questions generated. Try a different topic or increase sentence count.")
    else:
        st.error("Please upload a file and enter a topic.")
