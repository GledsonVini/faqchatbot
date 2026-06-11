"""
ShopBot Web UI — Streamlit interface for the RAG FAQ chatbot.
Run locally:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATASET_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
SAMPLE_SIZE = 2500  # keep embedding cost and cold-start time low on free hosting
TOP_K = 5

SYSTEM_PROMPT = """You are ShopBot, a friendly and professional customer support assistant for ShopEasy, a leading Canadian online retail platform. Answer customer questions accurately and warmly.

Use ONLY the context below. If the answer is not in the context, politely say you don't have that information and suggest contacting support@shopeasy.ca.

Context:
{context}
"""

st.set_page_config(page_title="ShopBot — RAG FAQ Chatbot", page_icon="🛍️")
st.title("🛍️ ShopBot — RAG FAQ Chatbot")
st.caption("Retrieval-Augmented Generation demo · Python · LangChain · FAISS · OpenAI")

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API key", type="password")
    if not api_key:
        st.info("Add an OpenAI API key in the sidebar to start chatting.")
        st.stop()


@st.cache_resource(show_spinner="Building knowledge base (first run only, ~1 min)...")
def build_index(key: str):
    df = pd.read_csv(DATASET_URL)
    df = df.dropna(subset=["instruction", "response"]).drop_duplicates(
        subset=["instruction", "response"]
    )
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    docs = [
        Document(
            page_content=f"Q: {row['instruction']}\nA: {row['response']}",
            metadata={
                "category": row.get("category", ""),
                "intent": row.get("intent", ""),
            },
        )
        for _, row in df.iterrows()
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=key)
    return FAISS.from_documents(chunks, embeddings)


vectorstore = build_index(api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about orders, refunds, shipping..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in retrieved)
    history = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages[:-1]
    ]
    chain = prompt | llm
    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.content
            for chunk in chain.stream(
                {"context": context, "history": history, "question": question}
            )
        )
    st.session_state.messages.append({"role": "assistant", "content": response})
