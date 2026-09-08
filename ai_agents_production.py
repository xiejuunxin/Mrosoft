import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

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
# 3. 定义 Expense 数据模型
# ============================================================

class Expense(BaseModel):
    date: str = Field(
        description="Date of expense in dd-MMM-yyyy format"
    )

    description: str = Field(
        description="Expense description"
    )

    amount: float = Field(
        description="Expense amount"
    )

    category: str = Field(
        description=(
            "Expense category, such as "
            "Transportation, Meals, Accommodation, Miscellaneous"
        )
    )


# ============================================================
# 4. ExpenseFormatter
# ============================================================

class ExpenseFormatter(BaseModel):

    raw_query: str = Field(
        description="Raw expense data"
    )

    def parse_expenses(self) -> list[Expense]:

        expense_list = []

        for expense_str in self.raw_query.split(";"):

            if not expense_str.strip():
                continue

            parts = expense_str.strip().split("|")

            if len(parts) != 4:
                continue

            date, description, amount, category = parts

            try:

                expense = Expense(
                    date=date.strip(),
                    description=description.strip(),
                    amount=float(amount.strip()),
                    category=category.strip()
                )

                expense_list.append(expense)

            except ValueError as e:

                print(
                    f"[LOG] Parse Error: "
                    f"Invalid data in '{expense_str}': {e}"
                )

        return expense_list


# ============================================================
# 5. 报销邮件 Tool
# ============================================================

@tool
def generate_expense_email(
    expense_data: str
) -> str:
    """
    Generate a professional expense claim email
    from semicolon-separated expense data.
    """

    formatter = ExpenseFormatter(
        raw_query=expense_data
    )

    expenses = formatter.parse_expenses()

    if not expenses:
        return "No valid expenses found."

    total_amount = sum(
        expense.amount
        for expense in expenses
    )

    email_body = "Dear Finance Team,\n\n"

    email_body += (
        "Please find below the details "
        "of my expense claim:\n\n"
    )

    for expense in expenses:

        email_body += (
            f"- {expense.date} | "
            f"{expense.description}: "
            f"${expense.amount:.2f} "
            f"({expense.category})\n"
        )

    email_body += (
        f"\nTotal Amount: "
        f"${total_amount:.2f}\n\n"
    )

    email_body += (
        "Receipts for all expenses "
        "are attached for your reference.\n\n"
    )

    email_body += "Thank you,\n[Your Name]"

    return email_body


# ============================================================
# 6. OCR Agent
# ============================================================

ocr_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
You are an expert OCR assistant specialized
in extracting structured data from receipt images.

Your task is to extract travel-related expenses.

For every expense, return:

date|description|amount|category

Multiple expenses must be separated by semicolons.

Rules:

- Date must use dd-MMM-yyyy format.
  Example: 04-Apr-2022

- Description should contain the item or expense name.

- Amount must be numeric.
  Example: 4.50

- Category should be one of:

  Transportation
  Accommodation
  Meals
  Miscellaneous

- Ignore totals and subtotals.

- Ignore service charges unless they
  are itemized expenses.

Return ONLY the structured expense data.

Example:

04-Apr-2022|Taxi|25.00|Transportation;
05-Apr-2022|Hotel|150.00|Accommodation
"""
)


# ============================================================
# 7. Email Agent
# ============================================================

email_agent = create_agent(
    model=model,
    tools=[generate_expense_email],
    system_prompt="""
You are an expense claim email generator.

You receive structured travel expense data
from another agent.

The input format is:

date|description|amount|category

Multiple expenses are separated by semicolons.

Your job is to call the
generate_expense_email tool.

Pass the expense data directly
to the tool.

Return the generated email.
"""
)


# ============================================================
# 8. OCR Agent
# ============================================================

def run_ocr_agent(receipt_text: str) -> str:

    result = ocr_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Extract the travel expenses
from the following receipt information:

{receipt_text}
"""
                }
            ]
        }
    )

    return result["messages"][-1].content


# ============================================================
# 9. Email Agent
# ============================================================

def run_email_agent(expense_data: str) -> str:

    result = email_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": expense_data
                }
            ]
        }
    )

    return result["messages"][-1].content


# ============================================================
# 10. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 模拟 OCR 获取的收据信息
    #
    # 实际项目中这里应该来自 receipt.jpg
    # --------------------------------------------------------

    receipt_text = """
    04/04/22
    Taxi Airport to Hotel
    $25.00

    04/04/22
    Hotel Room
    $150.00

    05/04/22
    Business Dinner
    $45.50
    """


    # ========================================================
    # Agent 1：OCR / Expense Extraction
    # ========================================================

    print("=" * 60)
    print("OCR AGENT")
    print("=" * 60)

    expense_data = run_ocr_agent(
        receipt_text
    )

    print(expense_data)


    # ========================================================
    # Agent 2：Email Generation
    # ========================================================

    print("\n" + "=" * 60)
    print("EMAIL AGENT")
    print("=" * 60)

    email = run_email_agent(
        expense_data
    )

    print(email)


# ============================================================
# 11. 程序入口
# ============================================================

if __name__ == "__main__":
    main()