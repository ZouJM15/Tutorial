import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime


load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")


class HooksDemo(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self.start_time = time.perf_counter()
        print("before_agent triggered")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("before_model triggered")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("after_model triggered")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        elapsed = time.perf_counter() - self.start_time
        print(f"after_agent triggered after {elapsed:.2f}s")
        return None


agent = create_agent(
    model=MODEL,
    middleware=[HooksDemo()],
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain PCA in one paragraph."}]}
)

print(response["messages"][-1].content)
