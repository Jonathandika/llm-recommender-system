import streamlit as st

from modules.RetrievalAugmentedGenerator_RL import RAG_RL
from modules.RecommendationSystem_RL import RecommendationSystemRL


with st.sidebar:
    "## CS4480 Group Project"
    "[View the source code](https://github.com/Jonathandika/llm-recommender-system)"
    "Authors: "
    "Jonathan Andika"
    "Seivabel Jessica"
    "Ryan Gani"

    # user_id = st.text_input("User ID", value="185")

st.title("💬 Chatbot")
st.caption("🚀 LLM Recommender System Chatbot")

rs = RecommendationSystemRL(retrain=False)
recommendation_agent = RAG_RL(rs, user_id=185)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = recommendation_agent.agent_invoke(prompt)
    
    msg = {"role": "assistant", "content": response}

    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])