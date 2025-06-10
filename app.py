import streamlit as st
from rag_chain_recipes import ask_rag_chatbot

st.set_page_config(page_title="MealBot - Homemade Recipe Recommender", layout="centered")
st.title("🥗 MealBot: Get Homemade Recipe Suggestions Based on Ingredients and Diet")

diet = st.selectbox("🔍 Select a dietary preference:", ["none", "vegetarian", "vegan", "Non-Vegetarian"])

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(("assistant", "👋 Hello! Tell me what ingredients you have, and I’ll suggest a recipe."))

if not st.session_state.chat_started:
    query = st.chat_input("🍳 What would you like to make?")
    if query:
        with st.spinner("Thinking of delicious recipes from our recipe book..."):
            full_query = f"My query: {query} and diet preference: {diet}"
            answer = ask_rag_chatbot(full_query)
            st.markdown(query)
        # with st.chat_message("assistant"):
        #     st.markdown(answer)
        st.session_state.chat_started = True
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()
else:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    query = st.chat_input("🍳 Any other questions?")
    if query:
        with st.spinner("Thinking..."):
            full_query = f"My query related to the chat history: {query}"
            answer = ask_rag_chatbot(full_query)
            st.markdown(query)
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()

if st.button("🗑️ Reset Chat"):
    st.session_state.chat_started = False
    st.session_state.chat_history = []
    st.rerun()


