import streamlit as st
from rag_pipeline import graph

st.set_page_config(page_title="Conversational RAG - Pedagogia Viva", layout="centered")

st.title("👩‍🏫 Assistente Pedagogico AI (Italiano/English)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages (Q&A pairs)
for entry in st.session_state.chat_history:
    q, a = entry.split("\nA: ")

    with st.chat_message("user"):
        st.markdown(q[3:])  # remove "Q: "

    with st.chat_message("assistant"):
        st.markdown(a)

# Get new input
user_input = st.chat_input("✍️ Fai la tua domanda (in italiano o inglese)")

if user_input:
    state = {
        "question": user_input,
        "chat_history": st.session_state.chat_history
    }

    with st.chat_message("user"):
        st.markdown(user_input)

    result = graph.invoke(state)

    with st.chat_message("assistant"):
        st.markdown(result["answer"])

    st.session_state.chat_history = result["chat_history"]
