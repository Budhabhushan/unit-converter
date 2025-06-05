import streamlit as st

st.title("Welcome to GOD-ChatGPT")

user_input = st.text_input("Play with me:")

st.write("You entred:", user_input)