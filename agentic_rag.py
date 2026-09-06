import os

from dotenv import load_dotenv
from typing import Annotated

from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain.agents import create_agent

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ==================================================
# 1. 加载环境变量
# ==================================================

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env")


# ==================================================
# 2. 创建 DeepSeek 模型
# ==================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key,
)


# ==================================================
# 3. 准备旅游知识
# ==================================================

documents = [
    Document(
        page_content=(
            "Barcelona is Spain's cosmopolitan capital of Catalonia. "
            "Best visited Mar-May or Sep-Nov. "
            "Known for Gaudí architecture, La Rambla, beaches. "
            "Average daily cost: $150-200."
        ),
        metadata={"destination": "Barcelona"},
    ),

    Document(
        page_content=(
            "Tokyo is Japan's capital, mixing ultramodern with traditional. "
            "Best visited Mar-Apr (cherry blossoms) or Oct-Nov. "
            "Known for Shibuya, temples, sushi. "
            "Average daily cost: $200-250."
        ),
        metadata={"destination": "Tokyo"},
    ),

    Document(
        page_content=(
            "Paris is France's capital and a global center for art, "
            "fashion, and culture. Best visited Apr-Jun or Sep-Oct. "
            "Known for Eiffel Tower, Louvre, cuisine. "
            "Average daily cost: $180-250."
        ),
        metadata={"destination": "Paris"},
    ),

    Document(
        page_content=(
            "Cape Town sits on South Africa's southwest tip. "
            "Best visited Nov-Mar. "
            "Known for Table Mountain, wine regions, wildlife. "
            "Average daily cost: $100-150."
        ),
        metadata={"destination": "Cape Town"},
    ),
]


# ==================================================
# 4. 创建 Embedding 模型
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 5. 创建 Chroma 向量数据库
# ==================================================

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="travel_knowledge",
    persist_directory="./chroma_db",
)


# ==================================================
# 6. 创建 Retriever
# ==================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ==================================================
# 7. 创建搜索 Tool
# ==================================================

@tool
def search_travel_knowledge(
    query: Annotated[
        str,
        "Search query about travel destinations"
    ],
) -> str:
    """Search the Chroma travel knowledge base."""

    docs = retriever.invoke(query)

    if not docs:
        return "No matching destinations found."

    results = []

    for doc in docs:
        destination = doc.metadata.get(
            "destination",
            "Unknown"
        )

        results.append(
            f"Destination: {destination}\n"
            f"Information: {doc.page_content}"
        )

    return "\n\n".join(results)


# ==================================================
# 8. 创建 Travel Agent
# ==================================================

travel_agent = create_agent(
    model=model,
    tools=[search_travel_knowledge],
    system_prompt="""You are a knowledgeable travel advisor.

Before answering questions about destinations:

1. ALWAYS search the travel knowledge base first.
2. Base your answers on retrieved information.
3. If information is not in the knowledge base, say so clearly.
4. Provide specific details such as costs, best seasons,
   and highlights.
""",
)


# ==================================================
# 9. 创建 Checker Agent
# ==================================================

checker_agent = create_agent(
    model=model,
    tools=[search_travel_knowledge],
    system_prompt="""You are a meticulous travel advisor.

When answering travel questions:

1. Search for relevant destinations first.
2. For each destination found, search again with the
   destination name to get full details.
3. Compare the options using retrieved information.
4. Present a final recommendation with costs,
   best travel times, and highlights.
5. Do not invent information that is not present
   in the knowledge base.
""",
)


# ==================================================
# 10. 运行 Agent
# ==================================================

def main():

    response = travel_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I'm interested in visiting somewhere "
                        "with great architecture. "
                        "What destinations would you recommend?"
                    ),
                }
            ]
        }
    )

    print("=== Travel Agent ===")
    print(response["messages"][-1].content)


    response = checker_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I have a $175/day budget and want "
                        "to travel in April. Which destinations "
                        "fit my budget and timing?"
                    ),
                }
            ]
        }
    )

    print("\n=== Checker Agent ===")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()