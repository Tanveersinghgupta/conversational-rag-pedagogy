import streamlit as st
from rag_pipeline import graph

st.set_page_config(page_title="Conversational RAG - Pedagogia Viva", layout="centered")

st.title("👩‍🏫 Assistente Pedagogico AI (Italiano/English)")

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_area("✍️ Fai la tua domanda (in italiano o inglese):", height=100)

if st.button("Invia"):
    if user_question.strip() != "":
        state = {
            "question": user_question,
            "chat_history": st.session_state.chat_history
        }
        result = graph.invoke(state)
        st.session_state.chat_history = result["chat_history"]
        st.markdown(f"**🧠 Risposta:** {result['answer']}")
    else:
        st.warning("Per favore, scrivi una domanda.")
