import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# 1. 初始化
# ============================================================

load_dotenv()

DEMO_MODE = True

# 每次运行生成一个独立的日志文件
GATE_LOG_PATH = Path(
    f"gate_log_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Please set it in your .env file."
    )


# LangChain + DeepSeek
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key,
)


# ============================================================
# 2. Pre-Action Gate：执行前审批
# ============================================================

def gate_action(
    action_description: str,
    risk_tier: str,
    attempt: int = 0,
) -> dict:

    """
    执行一次 Action Gate。

    decision:
        approve  -> 执行
        deny     -> 拒绝
        escalate -> 升级人工处理
    """

    print(
        f"[gate] proposed action "
        f"({risk_tier}, attempt={attempt}): "
        f"{action_description}"
    )

    # --------------------------------------------------------
    # Demo 模式
    # --------------------------------------------------------

    if DEMO_MODE:

        if risk_tier == "high":

            # 第一次拒绝
            # 第二次自动批准
            decision = "approve" if attempt >= 1 else "deny"

            reason = (
                "DEMO_MODE: scripted approval on retry"
                if attempt >= 1
                else "DEMO_MODE: high risk denied on first attempt"
            )

        else:

            decision = "approve"
            reason = (
                f"DEMO_MODE canned response "
                f"for tier={risk_tier}"
            )

    # --------------------------------------------------------
    # 真实人工审批
    # --------------------------------------------------------

    else:

        try:
            raw = input(
                "[gate] approve / deny / escalate? "
            ).strip().lower()

        except EOFError:
            raw = ""

        if raw in {"approve", "deny", "escalate"}:

            decision = raw
            reason = "operator input"

        elif raw == "":

            decision = "deny"
            reason = "no input received, defaulted to deny"

        else:

            decision = "deny"
            reason = (
                f"invalid input {raw!r}, "
                "defaulted to deny"
            )

    return {
        "decision": decision,
        "reason": reason,
        "action": action_description,
        "risk_tier": risk_tier,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# 3. Risk Tiering：风险分级
# ============================================================

LOW_RISK_KEYWORDS = {
    "look",
    "lookup",
    "search",
    "fetch",
    "read",
    "query",
    "view",
    "get",
    "list",
    "weather",
    "summarize",
}


HIGH_RISK_KEYWORDS = {
    "send",
    "email",
    "post",
    "publish",
    "charge",
    "pay",
    "transfer",
    "delete",
    "drop",
    "cancel",
    "refund",
}


MEDIUM_RISK_KEYWORDS = {
    "cache",
    "schedule",
    "reminder",
    "book",
    "reserve",
    "update",
    "increment",
    "log",
}


AUTO_APPROVE_REASONS = {
    "low": "auto-approved (low risk)",
    "medium": "auto-approved (medium risk, queued for batched review)",
}


def classify_risk(action: str) -> str:

    """
    根据关键词判断 Action 的风险等级。

    high -> 高风险
    low  -> 低风险
    medium -> 中风险

    未识别的操作默认 medium。
    """

    text = action.lower()

    # 高风险优先
    if any(
        keyword in text
        for keyword in HIGH_RISK_KEYWORDS
    ):
        return "high"

    # 低风险
    if any(
        keyword in text
        for keyword in LOW_RISK_KEYWORDS
    ):
        return "low"

    # 中风险
    if any(
        keyword in text
        for keyword in MEDIUM_RISK_KEYWORDS
    ):
        return "medium"

    # Fail-safe
    return "medium"


# ============================================================
# 4. Tiered Gate
# ============================================================

def tiered_gate(
    action: str,
    attempt: int = 0,
) -> dict:

    """
    先进行风险分类，再决定是否需要人工审批。
    """

    tier = classify_risk(action)

    # low / medium 自动批准
    if tier in AUTO_APPROVE_REASONS:

        return {
            "decision": "approve",
            "reason": AUTO_APPROVE_REASONS[tier],
            "action": action,
            "risk_tier": tier,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    # high -> 进入人工审批
    return gate_action(
        action,
        tier,
        attempt=attempt,
    )


# ============================================================
# 5. Audit Log：审计日志
# ============================================================

def log_decision(decision: dict) -> None:

    """
    将每次 Gate 决策写入 JSONL 文件。
    """

    with GATE_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                decision,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# 6. LLM Action Planner
# ============================================================

def propose_action(
    goal: str,
    prior_rejection: str | None = None,
) -> str:

    """
    使用 DeepSeek 生成一个具体的下一步 Action。
    """

    system_prompt = """
You are an action planner for an agent.

Your job is to propose ONE concrete next action
toward the user's goal.

Return only a single sentence.

If a previous proposal was rejected,
propose a different action that addresses
the rejection reason.
"""

    user_text = f"Goal: {goal}"

    if prior_rejection:

        user_text += (
            "\n\nPrevious proposal was denied.\n"
            f"Reason: {prior_rejection}\n"
            "Please propose a revised action."
        )

    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]
    )

    return response.content.strip()


# ============================================================
# 7. Revision Loop
# ============================================================

def run_with_revision(
    goal: str,
    max_revisions: int = 2,
) -> dict:

    """
    完整流程：

    LLM 提出 Action
          ↓
    风险分类
          ↓
    Gate
          ↓
    approve -> 结束
    deny    -> 重新规划
    """

    prior_reason = None

    for attempt in range(max_revisions + 1):

        # ----------------------------------------------------
        # Step 1：LLM 规划 Action
        # ----------------------------------------------------

        action = propose_action(
            goal,
            prior_rejection=prior_reason,
        )

        print(f"[planner] {action}")

        # ----------------------------------------------------
        # Step 2：风险分类 + Gate
        # ----------------------------------------------------

        decision = tiered_gate(
            action,
            attempt=attempt,
        )

        decision["attempt"] = attempt

        # ----------------------------------------------------
        # Step 3：写入审计日志
        # ----------------------------------------------------

        log_decision(decision)

        # ----------------------------------------------------
        # Step 4：判断
        # ----------------------------------------------------

        if decision["decision"] == "approve":

            return decision

        # ----------------------------------------------------
        # Step 5：拒绝 -> 把原因交给 LLM
        # ----------------------------------------------------

        prior_reason = decision["reason"]

    return {
        **decision,
        "final": "max_revisions_reached",
    }


# ============================================================
# 8. Demo
# ============================================================

goals = [
    "Look up the weather in Seattle for the customer's trip planning.",

    "Schedule a reminder for the customer to check in 24 hours before their flight.",

    "Send a marketing email to the customer about premium upgrade options.",
]


for goal in goals:

    print(f"\n=== Goal: {goal} ===")

    outcome = run_with_revision(
        goal,
        max_revisions=1,
    )

    print(
        f"[final] "
        f"{outcome['decision']} "
        f"({outcome['reason']})"
    )


# ============================================================
# 9. 查看 Audit Log
# ============================================================

print(
    f"\n=== Audit log "
    f"({GATE_LOG_PATH.name}) ==="
)

for line in GATE_LOG_PATH.read_text(
    encoding="utf-8"
).splitlines():

    record = json.loads(line)

    print(
        f"  [{record['risk_tier']:6s}] "
        f"{record['decision']:8s} "
        f"attempt={record.get('attempt', '?')} "
        f"action={record['action'][:140]}"
    )