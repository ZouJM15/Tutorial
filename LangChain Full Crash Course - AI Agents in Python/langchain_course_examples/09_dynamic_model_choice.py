import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage


load_dotenv()

BASIC_MODEL = os.getenv("BASIC_MODEL", "openai:gpt-4o-mini")
ADVANCED_MODEL = os.getenv("ADVANCED_MODEL", "openai:gpt-4.1-mini")

basic_model = init_chat_model(BASIC_MODEL, temperature=0.1)
advanced_model = init_chat_model(ADVANCED_MODEL, temperature=0.1)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.messages)
    model = advanced_model if message_count > 3 else basic_model
    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection],
)

short_response = agent.invoke(
    {
        "messages": [
            SystemMessage("You are a helpful assistant."),
            HumanMessage("What is one plus one?"),
        ]
    }
)

long_response = agent.invoke(
    {
        "messages": [
            SystemMessage("You are a helpful assistant."),
            HumanMessage("What is Python?"),
            AIMessage("Python is a programming language."),
            HumanMessage("What makes it popular?"),
        ]
    }
)

for label, response in [("short", short_response), ("long", long_response)]:
    last_message = response["messages"][-1]
    metadata = last_message.response_metadata or {}
    print(f"\n[{label}]")
    print(last_message.content)
    print("model:", metadata.get("model_name") or metadata.get("model"))
