"""
ShopBot: FAQ Chatbot for ShopEasy Online Retail
================================================
Uses LangChain + OpenAI GPT + FAISS for Retrieval-Augmented Generation (RAG)
Dataset: Bitext Customer Support LLM Chatbot Training Dataset (Kaggle)

Requirements:
    pip install langchain langchain-openai langchain-community langchain-text-splitters
    pip install faiss-cpu openai pandas tiktoken python-dotenv colorama
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from colorama import Fore, Style, init

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

init(autoreset=True)

DATASET_PATH  = "bitext_customer_support.csv"
FAISS_INDEX   = "shopbot_faiss_index"
TOP_K_RESULTS = 5
CHUNK_SIZE    = 450
CHUNK_OVERLAP = 60

SYSTEM_PROMPT = """You are ShopBot, a friendly and professional customer support assistant for ShopEasy, a leading Canadian online retail platform. Answer customer questions accurately and warmly.

Use ONLY the context below. If the answer is not in the context, politely say you don't have that information and suggest contacting support@shopeasy.ca.

Context:
{context}
"""

def load_and_preprocess(filepath):
    print(f"{Fore.CYAN}Loading dataset: {filepath}{Style.RESET_ALL}")
    df = pd.read_csv(filepath)
    df.dropna(subset=["instruction", "response"], inplace=True)
    df["instruction"] = df["instruction"].str.strip()
    df["response"] = df["response"].str.strip()
    if "category" in df.columns:
        df["category"] = df["category"].str.replace("_", " ").str.title()
    if "intent" in df.columns:
        df["intent"] = df["intent"].str.replace("_", " ").str.title()
    before = len(df)
    df.drop_duplicates(subset=["instruction", "response"], inplace=True)
    print(f"  Records after dedup: {len(df)} (removed {before - len(df)})")
    documents = []
    for _, row in df.iterrows():
        content = f"Q: {row['instruction']}\nA: {row['response']}"
        metadata = {"category": row.get("category", "General"), "intent": row.get("intent", "Unknown")}
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"  Documents created: {len(documents)}")
    return documents

def build_vector_store(documents, embeddings):
    print(f"{Fore.CYAN}Chunking and embedding...{Style.RESET_ALL}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks created")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX)
    print(f"  Index saved to '{FAISS_INDEX}/'")
    return vectorstore

def load_vector_store(embeddings):
    print(f"{Fore.CYAN}Loading FAISS index...{Style.RESET_ALL}")
    return FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(chat_history):
    messages = []
    for human, ai in chat_history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages

def build_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K_RESULTS})
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            chat_history=lambda x: format_history(x.get("chat_history", [])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

BANNER = f"""
{Fore.BLUE}{'='*60}
  ShopBot - ShopEasy Customer Support Assistant
  quit=exit | clear=reset | help=examples
{'='*60}{Style.RESET_ALL}
"""

EXAMPLES = [
    "How do I track my order?",
    "What is your return policy?",
    "How can I cancel my order?",
    "My payment was declined. What should I do?",
]

def run_chatbot(chain, retriever):
    print(BANNER)
    chat_history = []
    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}ShopBot: Goodbye!{Style.RESET_ALL}")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print(f"{Fore.YELLOW}ShopBot: Have a great day! 👋{Style.RESET_ALL}")
            break
        if user_input.lower() == "clear":
            chat_history = []
            print(f"{Fore.YELLOW}ShopBot: History cleared.{Style.RESET_ALL}\n")
            continue
        if user_input.lower() == "help":
            for q in EXAMPLES:
                print(f"  - {q}")
            print()
            continue
        try:
            answer = chain.invoke({"question": user_input, "chat_history": chat_history})
            print(f"\n{Fore.BLUE}ShopBot:{Style.RESET_ALL} {answer}\n")
            sources = retriever.invoke(user_input)
            if sources:
                cats = list({doc.metadata.get("category", "General") for doc in sources})
                print(f"{Fore.LIGHTBLACK_EX}  [Topics: {', '.join(cats[:3])}]{Style.RESET_ALL}\n")
            chat_history.append((user_input, answer))
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}\n")

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"{Fore.RED}ERROR: OPENAI_API_KEY not found in .env{Style.RESET_ALL}")
        sys.exit(1)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=512)
    if os.path.exists(FAISS_INDEX):
        vectorstore = load_vector_store(embeddings)
    else:
        if not os.path.exists(DATASET_PATH):
            print(f"{Fore.RED}ERROR: Dataset not found: '{DATASET_PATH}'{Style.RESET_ALL}")
            sys.exit(1)
        documents = load_and_preprocess(DATASET_PATH)
        vectorstore = build_vector_store(documents, embeddings)
    print(f"{Fore.CYAN}Building chain...{Style.RESET_ALL}")
    chain, retriever = build_chain(vectorstore, llm)
    print(f"{Fore.GREEN}ShopBot is ready!{Style.RESET_ALL}")
    run_chatbot(chain, retriever)

if __name__ == "__main__":
    main()