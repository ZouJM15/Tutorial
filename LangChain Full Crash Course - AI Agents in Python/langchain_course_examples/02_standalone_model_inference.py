import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")

model = init_chat_model(MODEL, temperature=0.1)
response = model.invoke("Hello, what is Python?")

print(response.content)
