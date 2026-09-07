import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. 环境变量
# ============================================================

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError(
        "Missing DEEPSEEK_API_KEY. "
        "Please set it in your .env file."
    )


# ============================================================
# 2. 创建 DeepSeek 模型
# ============================================================

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key,
)


# ============================================================
# 3. 定义结构化的旅行计划
# ============================================================

class TravelSubTask(BaseModel):
    task_id: int = Field(
        description="Unique ID of the subtask"
    )

    description: str = Field(
        description="Description of the subtask"
    )

    assigned_agent: str = Field(
        description=(
            "Agent responsible for this task, "
            "such as flight_agent, hotel_agent, "
            "or activity_agent"
        )
    )

    priority: str = Field(
        description="Priority: high, medium, or low"
    )

    dependencies: list[int] = Field(
        default_factory=list,
        description="IDs of tasks that must be completed first"
    )


class TravelPlan(BaseModel):
    destination: str

    trip_duration_days: int

    subtasks: list[TravelSubTask]

    total_estimated_budget_usd: int

    notes: str


# ============================================================
# 4. Planning Agent
# ============================================================

planning_agent = create_agent(
    model=model,

    system_prompt="""
You are a travel planning agent.

When given a travel request:

1. Break it into specific subtasks:
   flights, hotels, activities, and logistics.

2. Assign each subtask to the appropriate
   specialist agent.

3. Set priorities:
   high, medium, or low.

4. Identify dependencies between tasks.

5. Estimate the total budget.

Return the result using the required
TravelPlan structured format.
""",

    response_format=TravelPlan,
)


# ============================================================
# 5. 生成旅行计划
# ============================================================

result = planning_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Plan a 7-day trip to Paris for a couple "
                    "interested in art, cuisine, and history. "
                    "Budget around $5000."
                ),
            }
        ]
    }
)


# LangChain Structured Output
plan = result["structured_response"]


print(f"Destination: {plan.destination}")
print(f"Duration: {plan.trip_duration_days} days")
print(f"Budget: ${plan.total_estimated_budget_usd}")

print("\nSubtasks:")

for task in plan.subtasks:
    print(
        f"  [{task.priority}] "
        f"{task.task_id}. "
        f"{task.description} "
        f"→ {task.assigned_agent}"
    )


# ============================================================
# 6. 定义 Tools
# ============================================================

@tool
def book_flight(
    destination: str,
    departure_date: str,
    return_date: str,
) -> str:
    """
    Search and book flights for the trip.
    """

    return (
        f"Flight booked to {destination}: "
        f"{departure_date} → {return_date}, "
        f"confirmation #FLT-{hash(destination) % 10000:04d}"
    )


@tool
def reserve_hotel(
    city: str,
    check_in: str,
    check_out: str,
    guests: int,
) -> str:
    """
    Reserve a hotel room in the destination city.
    """

    return (
        f"Hotel reserved in {city}: "
        f"{check_in} to {check_out} "
        f"for {guests} guests, "
        f"confirmation #HTL-{hash(city) % 10000:04d}"
    )


@tool
def book_activity(
    activity_name: str,
    date: str,
    participants: int,
) -> str:
    """
    Book a tour, museum visit, or other activity.
    """

    return (
        f"Activity booked: "
        f"{activity_name} on {date} "
        f"for {participants} people, "
        f"confirmation #ACT-{hash(activity_name) % 10000:04d}"
    )


# ============================================================
# 7. Concierge Agent
# ============================================================

concierge_agent = create_agent(
    model=model,

    tools=[
        book_flight,
        reserve_hotel,
        book_activity,
    ],

    system_prompt="""
You are a travel concierge executing
a structured travel plan.

Use the available tools to fulfil
each subtask.

Work through the subtasks in order
and respect dependencies.

Summarise the results when finished.
""",
)


# ============================================================
# 8. 把 Planning Agent 的结果交给 Concierge Agent
# ============================================================

subtask_lines = "\n".join(
    f"- [{task.priority}] "
    f"{task.task_id}. "
    f"{task.description} "
    f"(agent: {task.assigned_agent}, "
    f"deps: {task.dependencies})"
    for task in plan.subtasks
)


execution_prompt = (
    f"Execute the following travel plan for "
    f"{plan.destination} "
    f"({plan.trip_duration_days} days, "
    f"${plan.total_estimated_budget_usd} budget):\n\n"
    f"{subtask_lines}"
)


# ============================================================
# 9. 执行计划
# ============================================================

exec_result = concierge_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": execution_prompt,
            }
        ]
    }
)


print("\n=== Execution Result ===")

print(
    exec_result["messages"][-1].content
)