import streamlit as st

from modules.RetrievalAugmentedGenerator import RAG

recommendation_agent = RAG()

with st.sidebar:
    "## CS4480 Group Project"
    "[View the source code](https://github.com/Jonathandika/llm-recommender-system)"
    "Authors: "
    "Jonathan Andika"
    "Seivabel Jessica"
    "Ryan Gani"

st.title("💬 Chatbot")
st.caption("🚀 LLM Recommender System Chatbot")

import streamlit as st

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = recommendation_agent.agent(prompt)
    print("REPONSE ===>", response)
    print("KEYS ===>", response.keys())
    
    msg = {"role": "assistant", "content": response["output"]}

    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])