import os
import openai
import requests
from autogen.agentchat import AssistantAgent, UserProxyAgent
from autogen import GroupChat, GroupChatManager

from dotenv import load_dotenv

load_dotenv() 
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a tool function to query the RAG backend
def search_knowledge_base(query: str) -> str:
    params = {"q": query, "top_k": 1}
    response = requests.get("http://127.0.0.1:8000/documents/search", params=params)
    
    if response.status_code != 200:
        return "Error: Failed to query the knowledge base."

    documents = response.json()
    return documents[0]["content"] if documents else "No result found."

# Main support assistant
support_agent = AssistantAgent(
    name="SupportAgent",
    system_message="You help users resolve Windows issues. Strictly use the search_knowledge_base tool only and response should be 2 - 3 lines only",
    llm_config={"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}
)
support_agent.register_function(
    {"search_knowledge_base": search_knowledge_base}
)

# Critic agent
critic_agent = AssistantAgent(
    name="CriticAgent",
    system_message="You review support answers for correctness and completeness. Suggest improvements if needed and reponse should be 3 - 5 lines only.",
    llm_config={"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}
)

# User proxy
user_proxy = UserProxyAgent(
    name="User",
    system_message="You are a user facing a Windows issue.",
    human_input_mode="NEVER"
)

# Group chat setup
group_chat = GroupChat(
    agents=[user_proxy, support_agent, critic_agent],
    messages=[],
    max_round=5
)
manager = GroupChatManager(groupchat=group_chat)

# Start conversation
user_proxy.initiate_chat(
    recipient=manager,
    message="Blue Screen (BSOD) Errors. How do I fix it?"
)
