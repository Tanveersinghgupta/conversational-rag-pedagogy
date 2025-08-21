import os
import uuid
from typing import List
import streamlit as st
from typing_extensions import TypedDict

from langchain.schema import Document
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langdetect import detect

from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
import os
load_dotenv()  

st.set_page_config(page_title="RAG Chat (Multi-Thread)", page_icon="💬", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not set in environment. Set it before running.")

@st.cache_resource(show_spinner=True)
def load_retriever():
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    faiss_index = FAISS.load_local(
        "./faiss_index_store",
        embedding,
        allow_dangerous_deserialization=True
    )
    return faiss_index.as_retriever()

retriever = load_retriever()

@st.cache_resource(show_spinner=True)
def build_graph(groq_api_key: str):
    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    def build_system_prompt(lang: str, context: str, question: str) -> str:
        file_map = {
            "it": "system_prompt_it.txt",
            "en": "system_prompt_en.txt"
        }
        file_name = file_map.get(lang, "system_prompt_en.txt")
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            # Fallback minimal template if file missing
            template = (
                "You are a helpful assistant.\n\n"
                "{context}\n\n"
                "Question: {question}\nAnswer:"
            )
        return template.format(context=context, question=question)

    def retrieve_node(state: MessagesState):
        user_msgs = [m for m in state["messages"] if m.type == "human"]
        user_question = user_msgs[-1].content if user_msgs else ""
        try:
            lang = detect(user_question)
        except Exception:
            lang = "en"

        docs: List[Document] = retriever.get_relevant_documents(user_question)
        context_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
        sys_prompt = build_system_prompt(lang, context=context_text, question=user_question)
        return {"messages": [SystemMessage(content=sys_prompt)]}

    def generate_node(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [AIMessage(content=response.content)]}

    memory = InMemorySaver()
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile(checkpointer=memory)
    return graph

graph = build_graph(GROQ_API_KEY or "")


def new_thread_id() -> str:
    return f"thread-{uuid.uuid4().hex[:8]}"

def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        cid = uuid.uuid4().hex[:6]
        st.session_state.chats[cid] = {
            "name": "Chat 1",
            "thread_id": new_thread_id(),
            "history": []
        }
        st.session_state.active_chat_id = cid

def send_to_graph(thread_id: str, human_text: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=human_text)]}
    result = graph.invoke(inputs, config)
    answer = result["messages"][-1].content
    return answer

init_session_state()

with st.sidebar:
    st.header("💬 Chats")
    all_chat_ids = list(st.session_state.chats.keys())
    names = [st.session_state.chats[cid]["name"] for cid in all_chat_ids]
    idx = 0
    if st.session_state.active_chat_id in all_chat_ids:
        idx = all_chat_ids.index(st.session_state.active_chat_id)

    choice = st.radio(
        "Select a chat",
        options=all_chat_ids,
        format_func=lambda cid: st.session_state.chats[cid]["name"],
        index=idx
    )
    st.session_state.active_chat_id = choice

    if st.button("➕ New chat"):
        cid = uuid.uuid4().hex[:6]
        st.session_state.chats[cid] = {
            "name": f"Chat {len(st.session_state.chats)+1}",
            "thread_id": new_thread_id(),
            "history": []
        }
        st.session_state.active_chat_id = cid
        st.rerun()

    active = st.session_state.chats[st.session_state.active_chat_id]
    new_name = st.text_input("Rename chat", value=active["name"])
    if new_name and new_name != active["name"]:
        active["name"] = new_name

    st.caption(f"Thread ID: `{active['thread_id']}`")
    if st.button("♻️ Reset this chat (keep thread_id)"):
        active["history"] = []
        st.success("Cleared messages.")
    if st.button("🗑️ Delete chat"):
        if len(st.session_state.chats) > 1:
            del st.session_state.chats[st.session_state.active_chat_id]
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
            st.rerun()
        else:
            st.warning("Keep at least one chat.")


tab_labels = [st.session_state.chats[cid]["name"] for cid in st.session_state.chats]
tabs = st.tabs(tab_labels)

for tab, cid in zip(tabs, st.session_state.chats.keys()):
    with tab:
        chat = st.session_state.chats[cid]
        st.subheader(chat["name"])
        st.caption(f"Thread: `{chat['thread_id']}`")

        for msg in chat["history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if cid == st.session_state.active_chat_id:
            user_text = st.chat_input("Ask something…")
            if user_text:

                chat["history"].append({"role": "user", "content": user_text})
                with st.chat_message("user"):
                    st.markdown(user_text)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            answer = send_to_graph(chat["thread_id"], user_text)
                        except Exception as e:
                            answer = f"⚠️ Error: {e}"
                        st.markdown(answer)

                chat["history"].append({"role": "assistant", "content": answer})

