import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain.agents import create_agent


# =========================
# 1. 加载环境变量
# =========================

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env")


# =========================
# 2. 创建 DeepSeek 模型
# =========================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key,
)


# =========================
# 3. 定义 Tools
# =========================

@tool
def get_destinations() -> list[str]:
    """Get available vacation destinations."""
    return [
        "Barcelona",
        "Paris",
        "Berlin",
        "Tokyo",
        "Sydney",
        "New York City",
    ]


@tool
def check_availability(
    destination: Annotated[str, "The destination to check"],
) -> str:
    """Check booking availability for a destination."""

    availability = {
        "Barcelona": "Available - 3 spots left",
        "Paris": "Available",
        "Berlin": "Sold out",
        "Tokyo": "Available - 1 spot left",
        "Sydney": "Available",
        "New York City": "Available",
    }

    return availability.get(
        destination,
        "Unknown destination",
    )


@tool
def get_flight_info(
    origin: Annotated[str, "Origin airport code"],
    destination: Annotated[str, "Destination airport code"],
) -> str:
    """Get flight information between two airports."""

    flights = {
        "LHR-BCN": "BA 2042, Departs 08:30, Arrives 11:45, $350",
        "LHR-CDG": "AF 1081, Departs 09:15, Arrives 11:30, $280",
        "LHR-NRT": "JL 044, Departs 11:00, Arrives 07:00+1, $890",
    }

    return flights.get(
        f"{origin}-{destination}",
        f"No direct flights from {origin} to {destination}",
    )


travel_tools = [
    get_destinations,
    check_availability,
    get_flight_info,
]


# =========================
# 4. 定义结构化输出
# =========================

class BookingRecommendation(BaseModel):
    destination: str = Field(
        description="Recommended travel destination"
    )

    available: bool = Field(
        description="Whether the destination is available"
    )

    flight_details: str = Field(
        description="Flight information"
    )

    estimated_cost: int = Field(
        description="Estimated flight cost in USD"
    )


class TravelPlan(BaseModel):
    recommendations: list[BookingRecommendation]


# =========================
# 5. 创建 Agent
# =========================

agent = create_agent(
    model=model,
    tools=travel_tools,
    system_prompt=(
        "You are a travel agent. "
        "Use the available tools to find destinations, "
        "check availability, and get flight information. "
        "Recommend suitable destinations based on the user's request."
    ),
    response_format=TravelPlan,
)


# =========================
# 6. 运行 Agent
# =========================

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I want to fly from London Heathrow "
                    "to somewhere warm in Europe. "
                    "Check what's available."
                ),
            }
        ]
    }
)


# =========================
# 7. 获取结构化结果
# =========================

travel_plan = result["structured_response"]

print(travel_plan)