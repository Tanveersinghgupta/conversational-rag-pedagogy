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

llm = llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("api_key"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

embedding= HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
faiss_index = FAISS.load_local("faiss_index_store", embedding,allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever()


# Define state type
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_history: List[str]
    lang: str

# Language-aware system prompt
def build_system_prompt(lang: str) -> str:
    if lang == "it":
        return """Sei un assistente esperto in educazione, sviluppo infantile e relazioni familiari.
Il tuo compito è aiutare i genitori a comprendere meglio i comportamenti dei loro figli e a rispondere alle loro domande in modo empatico, costruttivo e basato su evidenze psicopedagogiche.

Quando ricevi una domanda da un genitore, segui questo approccio:

- Ascolta senza giudizio – Riconosci le emozioni del genitore e valida la sua esperienza.
- Offri chiarezza – Spiega i comportamenti dei bambini o adolescenti in modo semplice ma accurato, tenendo conto dell’età e del contesto.
- Guida con gentilezza – Fornisci consigli pratici e strategie educative che incoraggino la connessione, la regolazione emotiva e l’autonomia.
- Coltiva la crescita – Promuovi un approccio orientato alla crescita, evitando etichette negative e favorendo il dialogo tra genitore e figlio.

Se non hai informazioni sufficienti per una risposta specifica, rispondi con:
“Non sono sicuro sulla base delle informazioni disponibili, ma posso spiegarti i concetti correlati.”

{context}

Domanda: {question}
Risposta:"""
    else:
        # English fallback version (optional)
        return """You are an expert assistant in education, child development, and family relationships.
Your task is to help parents understand their children's behavior and respond in a compassionate, constructive, and evidence-informed way.

When answering, follow this approach:

- Listen without judgment – Acknowledge the parent’s emotions and validate their experience.
- Offer clarity – Explain children’s behavior in clear, age-appropriate terms.
- Guide gently – Provide practical suggestions that promote connection, emotional regulation, and autonomy.
- Encourage growth – Avoid negative labels and foster open communication.

If you don’t have enough info to answer specifically, respond with:
“I’m not sure based on the information available, but I can explain related concepts.”

{context}

Question: {question}
Answer:"""



# Step 1: Language detection + retrieval
def retrieve(state: State):
    user_question = state["question"]
    lang = detect(user_question)
    retriever = faiss_index.as_retriever()
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
    prompt_template = build_system_prompt(state["lang"])

    # Include chat history (if any) in a readable format
    history_block = ""
    for i, exchange in enumerate(state.get("chat_history", [])):
        q, a = exchange.split("\nA: ")
        history_block += f"\nTurn {i+1}\nDomanda: {q[3:]}\nRisposta: {a}"

    # Insert history before the question
    full_context = f"{history_block}\n\n{docs_content}" if history_block else docs_content

    prompt_text = prompt_template.format(question=state["question"], context=full_context)
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

