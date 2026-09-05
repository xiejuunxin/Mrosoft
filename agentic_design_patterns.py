import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool


# ============================================================
# 1. 加载 .env
# ============================================================

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_URL")

if not api_key:
    raise ValueError("Missing DEEPSEEK_API_KEY")

if not base_url:
    raise ValueError("Missing DEEPSEEK_URL")


# ============================================================
# 2. 创建 DeepSeek 模型
# ============================================================

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
)


# ============================================================
# 3. 创建 Travel Concierge Agent
# ============================================================

travel_agent = create_agent(
    model=model,
    name="TravelConcierge",
    system_prompt="""
You are a luxury travel concierge named Alex.

Your role is to:
1. Understand the traveler's preferences.
2. Provide personalized travel suggestions.
3. Mention visa requirements and the best travel seasons.

Be warm, professional and enthusiastic about travel.
""")


# ============================================================
# 4. 定义 Tool
# ============================================================

class DestinationRecommendation(BaseModel):
    destination: str = Field(description="Destination name")
    available: bool = Field(description="Whether the destination is available")
    best_season: str = Field(description="Best travel season")
    highlights: list[str] = Field(description="Main highlights")
    estimated_budget_usd: int = Field(
        description="Estimated budget in USD"
    )


@tool
def get_destination_details(
    destination: Annotated[str, "The destination to look up"]
) -> DestinationRecommendation:
    """Get structured details about a vacation destination."""

    details = {
        "Barcelona": DestinationRecommendation(
            destination="Barcelona",
            available=True,
            best_season="May-Jun",
            highlights=[
                "Beach",
                "Architecture",
                "Nightlife",
            ],
            estimated_budget_usd=2000,
        ),

        "Tokyo": DestinationRecommendation(
            destination="Tokyo",
            available=True,
            best_season="Mar-Apr",
            highlights=[
                "Culture",
                "Food",
                "Technology",
            ],
            estimated_budget_usd=2500,
        ),

        "Cape Town": DestinationRecommendation(
            destination="Cape Town",
            available=False,
            best_season="Nov-Mar",
            highlights=[
                "Nature",
                "Wine",
                "Adventure",
            ],
            estimated_budget_usd=1800,
        ),
    }

    return details.get(
        destination,
        DestinationRecommendation(
            destination=destination,
            available=False,
            best_season="Unknown",
            highlights=[],
            estimated_budget_usd=0,
        ),
    )


# ============================================================
# 5. Destination Expert Agent
# ============================================================

destination_agent = create_agent(
    model=model,
    name="DestinationExpert",
    system_prompt="""
You are a destination research specialist.

Your job is to:
1. Evaluate destinations based on traveler preferences.
2. Check availability using the provided tool.
3. Return a short ranked list with pros and cons.

Do NOT discuss flights, hotels or logistics.
""",
    tools=[get_destination_details],
)


# ============================================================
# 6. Logistics Planner Agent
# ============================================================

logistics_agent = create_agent(
    model=model,
    name="LogisticsPlanner",
    system_prompt="""
You are a travel logistics planner.

Your job is to:
1. Create a day-by-day itinerary.
2. Suggest flight and hotel options within the stated budget.
3. Mention visa requirements and travel insurance.

Do NOT recommend destinations.
The Destination Expert handles destination selection.
""",
)


# ============================================================
# 7. 主程序
# ============================================================

async def main():

    # --------------------------------------------------------
    # Agent 1：Travel Concierge
    # --------------------------------------------------------

    response = await travel_agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'd love a week-long vacation somewhere with "
                    "great food and history. Budget around $2500."
                ),
            }
        ]
    })

    print("=== Travel Concierge ===")
    print(response["messages"][-1].content)


    # --------------------------------------------------------
    # Agent 2：Destination Expert
    # --------------------------------------------------------

    dest_response = await destination_agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "I want a week of culture and food "
                    "for under $2500. Where should I go?"
                ),
            }
        ]
    })

    destination_result = dest_response["messages"][-1].content

    print("\n=== Destination Expert ===")
    print(destination_result)


    # --------------------------------------------------------
    # Agent 3：Logistics Planner
    # --------------------------------------------------------

    logistics_response = await logistics_agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
Create a one-week travel plan based on this recommendation:

{destination_result}
""",
            }
        ]
    })

    print("\n=== Logistics Planner ===")
    print(logistics_response["messages"][-1].content)


# ============================================================
# 8. 启动程序
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())

