# import os
# from dotenv import load_dotenv

# from langchain.agents import create_agent
# from langchain.tools import tool
# from langchain_deepseek import ChatDeepSeek


# # =========================
# # 1. 加载 .env
# # =========================

# load_dotenv()


# # =========================
# # 2. 创建 DeepSeek 模型
# # =========================

# model = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
# )


# # =========================
# # 3. 创建工具
# # =========================

# @tool
# def get_destinations() -> list[str]:
#     """Get a list of popular vacation destinations."""

#     return [
#         "Barcelona",
#         "Paris",
#         "Berlin",
#         "Tokyo",
#         "Sydney",
#         "New York City",
#         "Cairo",
#         "Cape Town",
#         "Rio de Janeiro",
#         "Bali",
#     ]


# # =========================
# # 4. 创建 Agent
# # =========================

# agent = create_agent(
#     model=model,
#     tools=[get_destinations],
#     system_prompt=(
#         "You are a helpful travel agent. "
#         "Help users find their perfect vacation destination "
#         "based on their preferences. "
#         "Use the get_destinations tool to see available destinations."
#     ),
# )


# # =========================
# # 5. 普通调用
# # =========================

# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": (
#                     "I'm looking for a warm beach destination. "
#                     "What do you recommend?"
#                 ),
#             }
#         ]
#     }
# )

# print("===== Agent回答 =====")

# print(result["messages"][-1].content)


# # =========================
# # 6. 流式输出
# # =========================

# print("\n===== Stream输出 =====")

# for chunk in agent.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Tell me about Tokyo as a travel destination",
#             }
#         ]
#     },
#     stream_mode="messages",
# ):

#     token, metadata = chunk

#     if token.content:
#         print(token.content, end="", flush=True)

# print()

