import streamlit as st
from summarizer import summarize_text

st.set_page_config(page_title="Text Summarizer")

st.title("📝 Text Summarizer using LLM")

text_input = st.text_area("Enter text to summarize:", height=300)

max_len = st.slider("Max summary length", 50, 300, 130)
min_len = st.slider("Min summary length", 10, 100, 30)

if st.button("Summarize"):
    with st.spinner("Generating summary..."):
        summary = summarize_text(
            text_input,
            max_length=max_len,
            min_length=min_len
        )
        st.subheader("Summary:")
        st.success(summary)