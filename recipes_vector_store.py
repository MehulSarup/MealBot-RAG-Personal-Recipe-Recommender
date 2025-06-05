from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from data_loader import load_recipes_data
import pickle
import os

base_dir = os.path.dirname(__file__)
pkl_file = os.path.join(base_dir, 'recipe_vector_store.pkl')

def create_vector_store(docs, save_path = pkl_file):
    embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedding = embed_model)
    print("vector store created. Saving pickle file")
    with open(save_path, "wb") as f:
        pickle.dump(vector_store, f)
    print("Pickle file saved")
    return vector_store

def load_vector_store(csv_path, path= pkl_file):
    if not os.path.exists(path):
        print("Vector store not found. Creating new vector store from CSV.")
        documents = load_recipes_data(csv_path)
        create_vector_store(documents)

    with open(path, "rb") as f:
        return pickle.load(f)