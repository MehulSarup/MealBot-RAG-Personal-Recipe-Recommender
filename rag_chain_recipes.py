from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from recipes_vector_store import load_vector_store
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv, find_dotenv

# Fetch the LLM API Key
_ = load_dotenv(find_dotenv())  # read local .env file
llm_key = os.environ['OPEN_ROUTER']

# Fetch the csv file path
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "RAW_recipes.csv")

def init_rag_chain():
    custom_prompt = """
    You are Mealbot, an expert recipe assistant.
    
    Answer the user Question given below and use the chat history and context if given to answer the question. If the context is not enough, then say - I'm sorry I don't know any recipes.
    Question: {question}
    
    Context: {context}
    
    Chat History : {chat_history}
    
    """
    chain_prompt = PromptTemplate.from_template(template=custom_prompt)

    # Object to connect with Open AI supported LLMs
    llm = ChatOpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = llm_key,
        model = "deepseek/deepseek-r1-0528-qwen3-8b:free",
        temperature=0.1,
    )

    return chain_prompt | llm

def ask_rag_chatbot(ques):
    vector_store = load_vector_store(csv_path)
    retriever = vector_store.as_retriever(search_type="mmr", k=3)
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = ChatMessageHistory()

    # This chain seems to be not that effective when it's needed to work with a custom prompt
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm = llm,
    #     retriever = retriever,
    #     chain_type = "stuff",
    #     memory = memory
    # )

    chain = init_rag_chain()

    relevant_docs = retriever.get_relevant_documents(ques)
    context = "\n\n".join([d.page_content for d in relevant_docs])
    # Debug - print("Context:", context)

    trimmed_hist = memory.messages[-4:]
    chat_hist = "\n".join(
        [f"User: {m.content}" if m.type == "human" else f"Meal bot: {m.content}" for m in trimmed_hist])

    answer = chain.invoke({"question": ques, "context": context, "chat_history": chat_hist})
    print(answer)
    memory.add_user_message(ques)
    memory.add_ai_message(answer.content)

    return answer.content

# To be used when we want to interact with chatbot in CLI
# while True:
#     ques = input("Enter your query: ")
#     if ques == "exit":
#         print("Thank you! See ya soon")
#         break
#
#     print("Mealbot: ", ask_rag_chatbot(ques))

