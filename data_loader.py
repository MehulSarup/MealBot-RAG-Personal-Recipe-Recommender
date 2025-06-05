import pandas as pd
from langchain_core.documents import Document
import tiktoken

def classify_diet(ingredients):
    ingredients = ingredients.lower()
    if any(i in ingredients for i in ["chicken", "beef", "pork", "fish", "shrimp", "egg"]):
        return "Non-Vegetarian"
    elif any(i in ingredients for i in ["milk", "cheese", "butter"]):
        return "vegetarian"
    elif all(i not in ingredients for i in ["meat", "chicken", "beef", "fish", "egg", "milk", "cheese", "butter"]):
        return "vegan"
    else:
        return "vegetarian"

def token_count(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    print(enc)
    return len(enc.encode(text))

def load_recipes_data(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=["name", "steps", "description", "ingredients"])
    #df = df.head(50)
    #print(df.shape)

    docs=[]
    for _, row in df.iterrows():
        diet = classify_diet(row["ingredients"])
        content = "Recipe: {} \n Diet: {} \n Ingredients: {} \n Instructions: {} \n Description: {}".format(row["name"], diet, row["ingredients"], row["steps"], row["description"])
        tokens = token_count(content)
        if tokens > 1000:
            print("Warning: Token is high. Consider Chunking for ", row["name"])
        metadata = {"Recipe": row["name"], "Diet": diet}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

#csv_path = "C:/Users/mehul/Documents/Datasets/RAW_recipes.csv"
#recipe_docs = load_data(csv_path)
#print(recipe_docs[0])

