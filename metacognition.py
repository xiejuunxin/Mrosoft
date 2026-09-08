import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. 加载环境变量
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
# 3. 航班查询工具
# ============================================================

@tool
def get_flight_times(destination: str) -> str:
    """Get available flight times from the primary flight system."""

    destination = destination.strip().title()

    flights = {
        "Paris": "Departures: 08:00, 12:30, 17:45 — from $350",
        "Tokyo": "Departures: 11:00, 23:30 — from $890",
        "Barcelona": "Departures: 07:15, 14:00, 19:30 — from $280",
    }

    if destination in flights:
        return flights[destination]

    return (
        f"404: No flights found for {destination} in primary system. "
        "Use the backup flight system."
    )


# ============================================================
# 4. 备用航班系统
# ============================================================

@tool
def get_flight_times_backup(destination: str) -> str:
    """Get available flight times from the backup flight system."""

    destination = destination.strip().title()

    backup_flights = {
        "Berlin": "Departures: 09:00, 16:00 — from $220",
        "Sydney": "Departures: 22:00 — from $1200",
        "New York City": (
            "Departures: 06:00, 10:30, 15:00, 20:00 — from $450"
        ),
    }

    return backup_flights.get(
        destination,
        f"No flights found for {destination} in any system."
    )


# ============================================================
# 5. 第一阶段：Flight Agent
# ============================================================

flight_agent = create_agent(
    model=model,
    tools=[
        get_flight_times,
        get_flight_times_backup,
    ],
    system_prompt="""
You are a flight booking agent.

Your job is to answer the user's flight questions.

Rules:

1. Always try the primary flight system first.

2. If the primary system returns a 404 error,
   use the backup flight system.

3. Always tell the user whether the primary or
   backup system was used.

4. Give the user clear and useful flight information.

5. Do not invent flight information.
""",
)


# ============================================================
# 6. Evaluator：评价 Agent 的回答
# ============================================================

evaluation_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
You are a strict quality evaluator.

Evaluate the flight agent's answer.

Check three things:

1. Completeness
   Did the answer fully answer the user's question?

2. Accuracy
   Did the answer use only information supported
   by the flight tools?

3. Helpfulness
   Is the answer clear and useful to a traveler?

Give:

Completeness: 1-5
Accuracy: 1-5
Helpfulness: 1-5

Then determine:

PASS

or

FAIL

If FAIL, explain exactly what should be improved.
""",
)


# ============================================================
# 7. Reflection Agent
# ============================================================

reflection_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
You are a reflection agent.

Your job is to analyze a flight agent's previous answer
and the evaluator's feedback.

Think about:

1. What did the agent do correctly?
2. What went wrong?
3. Why did the problem happen?
4. What strategy should the agent change?
5. What instructions should the agent follow
   on the next attempt?

Return a concise revised strategy.
""",
)


# ============================================================
# 8. 重新执行 Agent
# ============================================================

def run_flight_agent(question: str, strategy: str = ""):

    prompt = f"""
User question:

{question}

Current strategy:

{strategy}

Use the available flight tools to answer the user.

Follow the current strategy carefully.
"""

    result = flight_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
    )

    return result["messages"][-1].content


# ============================================================
# 9. Evaluator
# ============================================================

def evaluate_answer(question: str, answer: str):

    prompt = f"""
User Question:
{question}

Agent Answer:
{answer}

Evaluate this answer according to your rules.
"""

    result = evaluation_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
    )

    return result["messages"][-1].content


# ============================================================
# 10. Reflection
# ============================================================

def reflect(question: str, answer: str, evaluation: str):

    prompt = f"""
User Question:
{question}

Previous Answer:
{answer}

Evaluator Feedback:
{evaluation}

Reflect on the previous answer.

Explain:

- What went wrong?
- Why did it happen?
- How should the strategy change?

Then provide a revised strategy.
"""

    result = reflection_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
    )

    return result["messages"][-1].content


# ============================================================
# 11. 主程序
# ============================================================

def main():

    question = "What flights are available to Berlin?"

    print("=" * 60)
    print("USER")
    print("=" * 60)

    print(question)


    # --------------------------------------------------------
    # 第一次执行
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("1. AGENT → FIRST ANSWER")
    print("=" * 60)

    answer = run_flight_agent(question)

    print(answer)


    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("2. EVALUATOR")
    print("=" * 60)

    evaluation = evaluate_answer(
        question,
        answer
    )

    print(evaluation)


    # --------------------------------------------------------
    # 判断是否需要 Reflection
    # --------------------------------------------------------

    if "FAIL" in evaluation.upper():

        print("\n" + "=" * 60)
        print("3. REFLECTION")
        print("=" * 60)

        strategy = reflect(
            question,
            answer,
            evaluation
        )

        print(strategy)


        # ----------------------------------------------------
        # 重新执行
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("4. RE-EXECUTE")
        print("=" * 60)

        new_answer = run_flight_agent(
            question,
            strategy
        )

        print(new_answer)


        # ----------------------------------------------------
        # 再次评价
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("5. FINAL EVALUATION")
        print("=" * 60)

        final_evaluation = evaluate_answer(
            question,
            new_answer
        )

        print(final_evaluation)

    else:

        print("\nAnswer passed evaluation.")
        print("No reflection required.")


# ============================================================
# 12. 程序入口
# ============================================================

if __name__ == "__main__":
    main()