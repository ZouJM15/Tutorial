import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool


load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")


@tool(
    "get_weather",
    description="Return weather information for a given city.",
    return_direct=False,
)
def get_weather(city: str) -> dict:
    response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=20)
    response.raise_for_status()
    return response.json()


agent = create_agent(
    model=MODEL,
    tools=[get_weather],
    system_prompt=(
        "You are a helpful weather assistant. Be concise, accurate, and a bit "
        "humorous while remaining useful."
    ),
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather like in Vienna?",
            }
        ]
    }
)

print(response["messages"][-1].content)
