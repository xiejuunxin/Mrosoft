# # import asyncio

# # from dotenv import load_dotenv
# # from langchain.agents import create_agent
# # from langchain.tools import tool
# # from langchain_deepseek import ChatDeepSeek

# # load_dotenv()


# # @tool
# # def book_flight(date: str, location: str) -> str:
# #     """Book travel given location and date."""
# #     return f"Travel was booked to {location} on {date}"


# # model = ChatDeepSeek(
# #     model="deepseek-chat",
# #     temperature=0,
# # )

# # agent = create_agent(
# #     model=model,
# #     tools=[book_flight],
# #     system_prompt="Help the user book travel. Use the book_flight tool when ready.",
# # )


# # async def main():
# #     result = await agent.ainvoke(
# #         {
# #             "messages": [
# #                 {
# #                     "role": "user",
# #                     "content": "I'd like to go to New York on January 1, 2025",
# #                 }
# #             ]
# #         }
# #     )

# #     print(result["messages"][-1].content)


# # if __name__ == "__main__":
# #     asyncio.run(main())
# import os
# import asyncio

# from dotenv import load_dotenv
# from langchain.agents import create_agent
# from langchain.tools import tool
# from langchain_deepseek import ChatDeepSeek

# load_dotenv()


# # =========================
# # 1. 创建 DeepSeek 模型
# # =========================

# model = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
# )


# # =========================
# # 2. 定义数据检索工具
# # =========================

# @tool
# def retrieve_tool(query: str) -> str:
#     """Retrieve relevant sales data based on the query."""
    
#     # 这里暂时模拟数据库查询
#     return (
#         "Q4 Sales Data:\n"
#         "October: $120,000\n"
#         "November: $150,000\n"
#         "December: $180,000\n"
#         "Total: $450,000"
#     )


# # =========================
# # 3. 定义数据分析工具
# # =========================

# @tool
# def analyze_tool(data: str) -> str:
#     """Analyze the retrieved sales data and provide insights."""
    
#     # 这里暂时让工具直接返回一个简单结果
#     return (
#         "Q4 sales totaled $450,000. "
#         "Sales increased each month, with December being the strongest month."
#     )


# # =========================
# # 4. 创建 Data Retrieval Agent
# # =========================

# agent_retrieve = create_agent(
#     model=model,
#     tools=[retrieve_tool],
#     system_prompt=(
#         "You are a data retrieval agent. "
#         "Retrieve relevant data using the available tools."
#     ),
# )


# # =========================
# # 5. 创建 Data Analysis Agent
# # =========================

# agent_analyze = create_agent(
#     model=model,
#     tools=[analyze_tool],
#     system_prompt=(
#         "You are a data analysis agent. "
#         "Analyze the retrieved data and provide useful insights."
#     ),
# )


# # =========================
# # 6. 运行两个 Agent
# # =========================

# async def main():

#     # 第一个 Agent：获取数据
#     retrieval_result = await agent_retrieve.ainvoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": "Retrieve sales data for Q4",
#                 }
#             ]
#         }
#     )

#     # 获取第一个 Agent 的最终回答
#     retrieved_data = retrieval_result["messages"][-1].content

#     print("===== Data Retrieval Agent =====")
#     print(retrieved_data)


#     # 第二个 Agent：分析数据
#     analysis_result = await agent_analyze.ainvoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": f"Analyze this data:\n{retrieved_data}",
#                 }
#             ]
#         }
#     )

#     analysis = analysis_result["messages"][-1].content

#     print("\n===== Data Analysis Agent =====")
#     print(analysis)

# if __name__ == "__main__":
#     asyncio.run(main())