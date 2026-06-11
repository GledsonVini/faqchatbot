# ShopBot — FAQ Chatbot with RAG

**[🚀 Live demo → faqchatbot23.streamlit.app](https://faqchatbot23.streamlit.app/)**

A customer support FAQ chatbot for a fictional Canadian online retailer (ShopEasy), built with **Retrieval-Augmented Generation (RAG)**. Instead of relying on the LLM's memory, the bot retrieves the most relevant Q&A pairs from a real customer support dataset and uses them as grounded context — reducing hallucinations and keeping answers on-policy.

## How it works

```
User question → Embedding → FAISS similarity search (top 5)
             → Retrieved Q&A context → GPT (system prompt + context + chat history)
             → Grounded answer
```

1. **Ingestion** — loads and deduplicates the [Bitext Customer Support dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset) (~27k Q&A pairs), converts each pair into a LangChain `Document` with category/intent metadata.
2. **Chunking** — splits documents with `RecursiveCharacterTextSplitter` (450 chars, 60 overlap).
3. **Indexing** — embeds chunks with OpenAI embeddings and stores them in a local **FAISS** vector index (persisted to disk, so it only builds once).
4. **Retrieval + Generation** — on each question, retrieves the top-5 most similar chunks and passes them as context to **GPT** via a LangChain chain, with conversation memory for follow-up questions.
5. **Guardrails** — the system prompt restricts answers to retrieved context only; out-of-scope questions get a polite fallback instead of a hallucination.

## Stack

- **Python** · pandas
- **LangChain** (chains, prompts, memory)
- **OpenAI API** (embeddings + chat completion)
- **FAISS** (vector similarity search)

## Running locally

```bash
git clone https://github.com/GledsonVini/faqchatbot.git
cd faqchatbot
pip install -r requirements.txt

# add your key
cp .env.example .env   # then edit .env

python faq_chatbot.py        # CLI version
streamlit run app.py         # Web UI version
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset) and save it as `bitext_customer_support.csv` in the project root.

**.env.example**
```
OPENAI_API_KEY=your_key_here
```

**requirements.txt**
```
langchain
langchain-openai
langchain-community
langchain-text-splitters
faiss-cpu
openai
pandas
tiktoken
python-dotenv
colorama
```

## Example

```
You: How do I cancel my order?
ShopBot: You can cancel your order by going to "My Orders" in your account...

You: And if it already shipped?
ShopBot: If your order has already shipped, you can refuse delivery or request a return...
```

## What I learned

- How RAG grounds LLM answers in retrieved context and why that beats fine-tuning for FAQ use cases
- Trade-offs in chunk size / overlap and `top_k` retrieval quality
- Managing conversation memory so follow-up questions keep context
- Built with AI-assisted development — every line reviewed and understood

## Roadmap

- [x] Web UI (chat interface) — [live on Streamlit Cloud](https://faqchatbot23.streamlit.app/)
- [x] Streaming responses
- [ ] Evaluation set to measure retrieval accuracy
