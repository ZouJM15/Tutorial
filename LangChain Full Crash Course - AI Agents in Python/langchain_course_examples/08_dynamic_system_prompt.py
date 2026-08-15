import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt


load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")


@dataclass
class UserRole:
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    match request.runtime.context.user_role:
        case "expert":
            return (
                "You are a technical machine-learning tutor. Use precise terms, "
                "linear algebra language, and implementation details."
            )
        case "child":
            return (
                "You explain ideas to a child. Use very simple words and one "
                "concrete analogy."
            )
        case _:
            return (
                "You are a beginner-friendly programming tutor. Explain clearly "
                "without assuming advanced math."
            )


agent = create_agent(
    model=MODEL,
    middleware=[user_role_prompt],
    context_schema=UserRole,
)

for role in ["beginner", "expert", "child"]:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Explain PCA."}]},
        context=UserRole(user_role=role),
    )
    print(f"\n[{role}]")
    print(response["messages"][-1].content)
