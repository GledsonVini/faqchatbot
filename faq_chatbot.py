"""
ShopBot: FAQ Chatbot for ShopEasy Online Retail
================================================
Uses LangChain + OpenAI GPT + FAISS for Retrieval-Augmented Generation (RAG)
Dataset: Bitext Customer Support LLM Chatbot Training Dataset (Kaggle)

Requirements:
    pip install langchain langchain-openai langchain-community faiss-cpu
    pip install openai pandas tiktoken python-dotenv colorama
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from colorama import Fore, Style, init

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATASET_PATH   = "bitext_customer_support.csv"   # CSV downloaded from Kaggle
FAISS_INDEX    = "shopbot_faiss_index"            # Saved vector store path
TOP_K_RESULTS  = 5                                # Chunks retrieved per query
CHUNK_SIZE     = 450                              # Tokens per chunk
CHUNK_OVERLAP  = 60                               # Overlap between chunks

# System prompt injected into every conversation turn
SYSTEM_PROMPT = """You are ShopBot, a friendly and professional customer support \
assistant for ShopEasy, a leading Canadian online retail platform. Your role is to \
answer customer questions accurately, concisely, and warmly.

Use ONLY the context provided below to answer questions. If the answer is not in \
the context, politely say you don't have that information and suggest the customer \
contact ShopEasy support at support@shopeasy.ca or call 1-800-SHOP-NOW.

Never make up order numbers, policies, or prices. Stay on topic.

Context:
{context}
"""

# ---------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def load_and_preprocess(filepath: str) -> list[Document]:
    """
    Load the Bitext Customer Support CSV and convert each row into a
    LangChain Document object formatted as a Q&A pair.

    Preprocessing steps applied:
      1. Drop rows with null instruction or response fields.
      2. Strip leading/trailing whitespace from text fields.
      3. Normalize category and intent labels to title case.
      4. Deduplicate on (instruction, response) pairs.
      5. Format as 'Q: <instruction>\\nA: <response>' for retrieval.
    """
    print(f"{Fore.CYAN}Loading dataset from: {filepath}{Style.RESET_ALL}")

    df = pd.read_csv(filepath)

    # ---- Preprocessing ----
    # Step 1: Drop nulls in key columns
    df.dropna(subset=["instruction", "response"], inplace=True)

    # Step 2: Strip whitespace
    df["instruction"] = df["instruction"].str.strip()
    df["response"]    = df["response"].str.strip()

    # Step 3: Normalize labels
    if "category" in df.columns:
        df["category"] = df["category"].str.replace("_", " ").str.title()
    if "intent" in df.columns:
        df["intent"] = df["intent"].str.replace("_", " ").str.title()

    # Step 4: Deduplicate
    before = len(df)
    df.drop_duplicates(subset=["instruction", "response"], inplace=True)
    after = len(df)
    print(f"  Deduplication: {before} → {after} records ({before - after} removed)")

    # Step 5: Build LangChain Documents
    documents = []
    for _, row in df.iterrows():
        content = f"Q: {row['instruction']}\nA: {row['response']}"
        metadata = {
            "category": row.get("category", "General"),
            "intent":   row.get("intent",   "Unknown"),
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"  Total documents created: {len(documents)}")
    return documents


# ---------------------------------------------------------------------------
# VECTOR STORE (FAISS)
# ---------------------------------------------------------------------------

def build_vector_store(documents: list[Document], embeddings) -> FAISS:
    """
    Chunk documents and embed them into a FAISS vector store.
    Saves the index to disk for fast reloading.
    """
    print(f"{Fore.CYAN}Chunking documents...{Style.RESET_ALL}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks created from {len(documents)} documents")

    print(f"{Fore.CYAN}Embedding chunks and building FAISS index...{Style.RESET_ALL}")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX)
    print(f"  Index saved to '{FAISS_INDEX}/'")
    return vectorstore


def load_vector_store(embeddings) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    print(f"{Fore.CYAN}Loading existing FAISS index from '{FAISS_INDEX}/'...{Style.RESET_ALL}")
    return FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# CONVERSATIONAL CHAIN
# ---------------------------------------------------------------------------

def build_chain(vectorstore: FAISS, llm) -> ConversationalRetrievalChain:
    """
    Assemble a LangChain ConversationalRetrievalChain with:
      - FAISS retriever (top-K similarity search)
      - Conversation buffer memory (remembers last N turns)
      - Custom system prompt
    """
    # Custom prompts
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["context"], template=SYSTEM_PROMPT)
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["question"], template="{question}")
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ---------------------------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------------------------

BANNER = f"""
{Fore.BLUE}{'=' * 62}
  ____  _                 ____        _
 / ___|| |__   ___  _ __ | __ )  ___ | |_
 \___ \| '_ \ / _ \| '_ \|  _ \ / _ \| __|
  ___) | | | | (_) | |_) | |_) | (_) | |_
 |____/|_| |_|\___/| .__/|____/ \___/ \__|
                   |_|
{'=' * 62}
  ShopEasy Customer Support Assistant
  Type 'quit' or 'exit' to end the session.
  Type 'clear' to reset conversation history.
  Type 'help' to see example questions.
{'=' * 62}{Style.RESET_ALL}
"""

EXAMPLE_QUESTIONS = [
    "How do I track my order?",
    "What is your return policy?",
    "How can I cancel my order?",
    "My payment was declined. What should I do?",
    "How long does shipping take?",
    "Can I change my delivery address after ordering?",
]


def run_chatbot(chain: ConversationalRetrievalChain):
    """Main REPL loop for the chatbot."""
    print(BANNER)

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}ShopBot: Goodbye! Have a great day.{Style.RESET_ALL}")
            break

        if not user_input:
            continue

        # Special commands
        if user_input.lower() in ("quit", "exit", "bye"):
            print(f"{Fore.YELLOW}ShopBot: Thank you for contacting ShopEasy. "
                  "Have a wonderful day! 👋{Style.RESET_ALL}")
            break

        if user_input.lower() == "clear":
            chain.memory.clear()
            print(f"{Fore.YELLOW}ShopBot: Conversation history cleared.{Style.RESET_ALL}\n")
            continue

        if user_input.lower() == "help":
            print(f"{Fore.YELLOW}ShopBot: Here are some questions you can ask me:{Style.RESET_ALL}")
            for q in EXAMPLE_QUESTIONS:
                print(f"  • {q}")
            print()
            continue

        # Query the chain
        try:
            result = chain.invoke({"question": user_input})
            answer = result.get("answer", "I'm sorry, I couldn't generate a response.")

            print(f"\n{Fore.BLUE}ShopBot:{Style.RESET_ALL} {answer}\n")

            # Optional: Show source categories (for transparency / debugging)
            sources = result.get("source_documents", [])
            if sources:
                categories = list({doc.metadata.get("category", "General")
                                   for doc in sources})
                print(f"{Fore.LIGHTBLACK_EX}  [Sources: {', '.join(categories[:3])}]{Style.RESET_ALL}\n")

        except Exception as e:
            print(f"{Fore.RED}ShopBot: An error occurred: {e}{Style.RESET_ALL}\n")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    # Load environment variables (.env file with OPENAI_API_KEY)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"{Fore.RED}ERROR: OPENAI_API_KEY not found.{Style.RESET_ALL}")
        print("Create a .env file with: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Initialize models
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=512,
    )

    # Build or load vector store
    if os.path.exists(FAISS_INDEX):
        vectorstore = load_vector_store(embeddings)
    else:
        if not os.path.exists(DATASET_PATH):
            print(f"{Fore.RED}ERROR: Dataset not found at '{DATASET_PATH}'.{Style.RESET_ALL}")
            print("Download from: https://www.kaggle.com/datasets/bitext/"
                  "training-dataset-for-chatbot-and-nlu")
            sys.exit(1)
        documents = load_and_preprocess(DATASET_PATH)
        vectorstore = build_vector_store(documents, embeddings)

    # Build chain and start chatbot
    print(f"{Fore.CYAN}Initializing conversational chain...{Style.RESET_ALL}")
    chain = build_chain(vectorstore, llm)
    print(f"{Fore.GREEN}ShopBot is ready!{Style.RESET_ALL}")

    run_chatbot(chain)


if __name__ == "__main__":
    main()
