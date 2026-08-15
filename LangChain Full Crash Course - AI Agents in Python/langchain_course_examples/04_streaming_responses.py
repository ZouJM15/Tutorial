import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

MODEL = os.getenv("STREAM_MODEL", os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini"))

model = init_chat_model(MODEL, temperature=0.3)

for chunk in model.stream("Explain Python in three short paragraphs."):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()
