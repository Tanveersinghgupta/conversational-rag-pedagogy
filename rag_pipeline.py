from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import StateGraph, START
from langdetect import detect
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
import os
load_dotenv()  

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("api_key"),
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

embedding= HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
faiss_index = FAISS.load_local("faiss_index_store", embedding,allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 3, 'fetch_k': 50}
    )

# Define state type
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_history: List[str]
    lang: str
    
def build_system_prompt(lang: str, context: str, question: str) -> str:
    file_map = {
        "it": "system_prompt_it.txt",
        "en": "system_prompt_en.txt"
    }
    file_name = file_map.get(lang, "system_prompt_en.txt")
    
    with open(file_name, 'r', encoding='utf-8') as f:
        template = f.read()

    return template.format(context=context, question=question)


# Step 1: Language detection + retrieval
def retrieve(state: State):
    user_question = state["question"]
    lang = detect(user_question)
    retriever = faiss_index.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 50})
    docs = retriever.get_relevant_documents(user_question)
    return {
        "question": user_question,
        "context": docs,
        "chat_history": state.get("chat_history", []),
        "lang": lang
    }

# Step 2: Generation with memory + language-aware prompt
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    history_block = ""
    for i, exchange in enumerate(state.get("chat_history", [])):
        q, a = exchange.split("\nA: ")
        history_block += f"\nTurn {i+1}\nDomanda: {q[3:]}\nRisposta: {a}"

    # Insert history before the question
    full_context = f"{history_block}\n\n{docs_content}" if history_block else docs_content

    prompt_text =  build_system_prompt(state["lang"], context=full_context, question=state["question"])
    messages = [{"role": "system", "content": prompt_text}]
    response = llm.invoke(messages)

    updated_history = state["chat_history"] + [f"Q: {state['question']}\nA: {response.content}"]
    
    return {
        "answer": response.content,
        "chat_history": updated_history,
        "question": state["question"],
        "context": state["context"],
        "lang": state["lang"]
    }

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()