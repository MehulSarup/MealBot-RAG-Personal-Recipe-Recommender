from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from recipes_vector_store import load_vector_store
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv, find_dotenv

# Fetch the LLM API Key
_ = load_dotenv(find_dotenv()) # read local .env file
llm_key = os.environ['OPEN_ROUTER']

# Fetch the csv file path
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "RAW_recipes.csv")

custom_prompt = """
You are Mealbot, an expert recipe assistant.

Answer the user Question given below and use the chat history if given as context to answer the question.
Question: {question}
Chat History : {chat_history}
"""
chain_prompt = PromptTemplate.from_template(template=custom_prompt)

llm = ChatOpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = llm_key,
    model = "deepseek/deepseek-r1-0528-qwen3-8b:free",
    temperature=0.1,
)

vector_store = load_vector_store(csv_path)
retriever = vector_store.as_retriever(search_type="similarity", k=5)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    chain_type = "stuff",
    memory = memory
)

while True:
    ques = input("Enter your query: ")
    if ques == "exit":
        break
    answer = chain.invoke({"question": ques})
    print(answer["answer"])
