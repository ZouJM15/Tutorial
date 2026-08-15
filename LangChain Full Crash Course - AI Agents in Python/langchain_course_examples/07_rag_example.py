import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

similarity_texts = [
    "Apple makes very good computers.",
    "I believe Apple is innovative.",
    "I love apples.",
    "I am a fan of MacBooks.",
    "I enjoy oranges.",
    "I like Lenovo ThinkPads.",
    "I think pears taste very good.",
]

similarity_store = FAISS.from_texts(similarity_texts, embedding=embeddings)

print("Fruit query:")
for document in similarity_store.similarity_search("Apples are my favorite food.", k=7):
    print("-", document.page_content)

print("\nComputer query:")
for document in similarity_store.similarity_search("Linux is a great operating system.", k=7):
    print("-", document.page_content)

knowledge_texts = [
    "I love apples.",
    "I enjoy oranges.",
    "I think pears taste very good.",
    "I hate bananas.",
    "I dislike raspberries.",
    "I despise mangoes.",
    "I love Linux.",
    "I hate Windows.",
]

knowledge_store = FAISS.from_texts(knowledge_texts, embedding=embeddings)
retriever = knowledge_store.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    name="kb_search",
    description="Search a small personal preference database.",
)

agent = create_agent(
    model=CHAT_MODEL,
    tools=[retriever_tool],
    system_prompt=(
        "Answer questions from the knowledge base. Use kb_search before "
        "answering. If needed, search more than once. Keep the final answer "
        "short."
    ),
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Which three fruits does this person like, and which three fruits do they hate?",
            }
        ]
    }
)

print("\nAgent answer:")
print(response["messages"][-1].content)
