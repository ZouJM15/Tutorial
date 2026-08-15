import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage


load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")

model = init_chat_model(MODEL, temperature=0.1)

conversation = [
    SystemMessage("You are a helpful assistant for questions about programming."),
    HumanMessage("What is Python?"),
    AIMessage("Python is a high-level interpreted programming language."),
    HumanMessage("When was it first released?"),
]

response = model.invoke(conversation)

print(response.content)
