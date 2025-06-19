import os
import requests
import agentops
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.tools import Tool, tool

# Load environment variables
load_dotenv()

# Initialize AgentOps
agentops.init()

# Tool to query RAG backend
@tool
def query_rag(tool_input: str) -> str:
    """RAG BACKEND"""
    try:
        response = requests.get(
            "http://127.0.0.1:8000/document/search",
            params={"query": tool_input, "top_k": 3}, 
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and data:
            return "\n\n".join([item.get("content", "") for item in data])
        return "No relevant content found."
    except Exception as e:
        return f"Error calling RAG: {e}"

# Wrap the tool using LangChain's Tool class
query_rag_tool = Tool(
    name="query_rag",
    func=query_rag,
    description="Queries the RAG backend to fetch new Python features."
)

# User agent
user_agent = Agent(
    role="User",
    goal="Ask about the top 3 new Python features",
    backstory="You are a curious developer interested in Python updates.",
    verbose=True
)

# Support agent that retrieves answers
support_agent = Agent(
    role="Research Agent",
    goal="Find and summarize the top 3 new Python features",
    backstory="You are a technical assistant with access to documentation and tools like RAG.",
    tools=[query_rag_tool],  #  Correct format
    verbose=True
)

# Critic agent to validate the answer
critic_agent = Agent(
    role="Critic Agent",
    goal="Validate the summary for correctness and completeness",
    backstory="You are an expert Python reviewer ensuring factual accuracy and completeness.",
    verbose=True
)

# Define hierarchical tasks
tasks = [
    Task(description="What are the top 3 new features in Python?", agent=user_agent),
    Task(description="Use query_rag and summarize top 3 features from RAG", agent=support_agent),
    Task(description="Review the summary for accuracy", agent=critic_agent)
]

# Assemble crew with hierarchical sequential process
crew = Crew(
    agents=[user_agent, support_agent, critic_agent],
    tasks=tasks,
    process="sequential",
    verbose=True
)

# Run the crew
result = crew.kickoff()
print("\nFinal Result:\n", result)
