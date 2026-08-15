import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field


load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")


@dataclass
class Context:
    user_id: str


class WeatherResponse(BaseModel):
    summary: str = Field(description="A concise natural-language weather summary.")
    temperature_celsius: float | None = Field(default=None)
    temperature_fahrenheit: float | None = Field(default=None)
    humidity: float | None = Field(default=None)


@tool("get_weather", description="Return weather information for a given city.")
def get_weather(city: str) -> dict:
    response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=20)
    response.raise_for_status()
    return response.json()


@tool("locate_user", description="Look up a user's city based on runtime context.")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    match runtime.context.user_id:
        case "ABC123":
            return "Vienna"
        case "XYZ456":
            return "London"
        case "HJKL111":
            return "Paris"
        case _:
            return "unknown"


model = init_chat_model(MODEL, temperature=0.3)
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt=(
        "You are a helpful weather assistant. If the user does not provide a "
        "city, use locate_user first. Then use get_weather. Return structured "
        "weather data."
    ),
    context_schema=Context,
    response_format=WeatherResponse,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "demo-weather-thread"}}

first_response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather like?"}]},
    context=Context(user_id="ABC123"),
    config=config,
)

structured = first_response["structured_response"]
print(structured.summary)
print(f"Celsius: {structured.temperature_celsius}")

follow_up = agent.invoke(
    {"messages": [{"role": "user", "content": "Is that normal for this season?"}]},
    context=Context(user_id="ABC123"),
    config=config,
)

print(follow_up["messages"][-1].content)
