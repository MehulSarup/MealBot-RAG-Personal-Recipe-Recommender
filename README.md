# 🥗 MealBot: A RAG-Powered AI Recipe Recommender

MealBot is an AI-powered meal recommendation assistant built using **Retrieval-Augmented Generation (RAG)**. It helps users discover personalized recipes based on their available ingredients and dietary preferences, and supports contextual follow-up questions using conversation memory.

---

## 🎯 Project Goal

To demonstrate how modern LLMs and vector search can be combined to solve real-world problems like daily meal planning. The aim is to:
- Build a smart, interactive assistant using a **custom prompt-driven RAG pipeline**
- Integrate **cloud-hosted free LLMs** for cost-effective deployment
- Highlight key skills needed in modern **data science & AI engineer roles**

---

## 🧠 How It Works

- **Document Ingestion**: Loads structured recipe data from a CSV and enriches it with inferred dietary metadata.
- **Embedding & Indexing**: Embeds recipes using Sentence Transformers and stores them in a FAISS vector database.
- **Retrieval-Augmented Generation**:
  - Retrieves top relevant recipes based on user input using semantic vector search
  - Feeds retrieved context + chat history into an LLM with a custom prompt
- **Conversational Memory**: Uses LangChain's `ConversationBufferMemory` to support multi-turn chat.

---

## 🧰 Tech Stack

| Component | Tool |
|----------|------|
| Language Model | [`deepseek-r1` via OpenRouter API](https://openrouter.ai/) |
| Embedding Model | `all-MiniLM-L6-v2` (Hugging Face Transformers) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Chat Memory | ConversationBufferMemory |
| Deployment Ready | (Supports Streamlit frontend or terminal CLI) |

---

## 📊 Example Use Cases

- **Query**: "I have pasta, broccoli, and spinach. What can I cook?"
- **Follow-Up**: "Make it vegetarian."
- **Further**: "How long will it take to cook?"

💬 The assistant retrieves matching recipes and responds with grounded, natural explanations while keeping context.

---

## 📁 Project Structure
- data_loader.py # Recipe loading, cleaning, diet classification
- recipes_vector_store.py # FAISS vector store build/save
- rag_chain_recipes.py # LangChain RAG pipeline with custom prompt
- app.py # Streamlit (optional) or CLI chat loop
- recipes.csv # Input recipe dataset
- vector_store.pkl # Saved FAISS index
