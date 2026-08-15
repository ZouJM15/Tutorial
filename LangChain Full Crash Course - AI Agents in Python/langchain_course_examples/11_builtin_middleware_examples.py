import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware


load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai:gpt-4o-mini")

agent = create_agent(
    model=MODEL,
    middleware=[
        SummarizationMiddleware(
            model=SUMMARY_MODEL,
            max_tokens_before_summary=4000,
            messages_to_keep=20,
        )
    ],
)

messages = [
    {
        "role": "user",
        "content": (
            "We are planning a Python project that uses LangChain agents. "
            "Remember that the app needs weather tools, RAG over notes, and "
            "different explanations for beginners and experts."
        ),
    },
    {"role": "user", "content": "Summarize the project requirements briefly."},
]

response = agent.invoke({"messages": messages})

print(response["messages"][-1].content)

BUILT_IN_MIDDLEWARE_TO_EXPLORE = [
    "SummarizationMiddleware: summarize long conversations automatically.",
    "Human-in-the-loop middleware: pause risky tool calls for approval.",
    "Model call limits: cap model usage per run or thread.",
    "Tool call limits: cap tool usage per run or thread.",
    "Model fallback: try another model when the first one fails.",
    "PII middleware: detect or redact personally identifiable information.",
    "Tool retry middleware: retry failing tools with controlled behavior.",
]

print("\nOther middleware ideas from the video:")
for item in BUILT_IN_MIDDLEWARE_TO_EXPLORE:
    print("-", item)
