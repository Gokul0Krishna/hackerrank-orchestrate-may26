import os
import json
import anthropic
from retriever import Retriever

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
_retriever = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# Keywords that should trigger automatic escalation regardless of LLM output
HIGH_RISK_SIGNALS = [
    'fraud', 'unauthorized transaction', 'unauthorized charge', 'stolen card',
    'stolen account', 'hacked', 'compromised', 'security breach', 'chargeback',
    'dispute', 'legal action', 'lawsuit', 'refund', 'account suspended',
    'account banned', 'account blocked', 'account locked', 'double charged',
    'wrong charge', 'identity theft', 'phishing', 'scam', 'cannot access my account',
    'lost access', 'payment issue', 'card declined', 'data breach'
]

SYSTEM_PROMPT = """You are a support triage agent for three products: HackerRank, Claude (by Anthropic), and Visa.

Your task: analyze each support ticket and return a structured JSON decision.

Output exactly this JSON schema — no markdown, no extra text:
{
  "status": "replied" | "escalated",
  "product_area": "<string>",
  "response": "<user-facing response string>",
  "justification": "<internal reasoning string>",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid"
}

Field definitions:
- status: "replied" if you can answer safely from the corpus; "escalated" if human intervention needed
- product_area: the support category (e.g., billing, account_access, assessments, fraud, technical_issue, general_inquiry, canvas, coding_environment, subscriptions, card_security, etc.)
- response: what you say to the user — must be grounded in the provided corpus excerpts only
- justification: concise internal note explaining your decision (not shown to user)
- request_type: product_issue | feature_request | bug | invalid

Escalation rules (non-negotiable):
- Always escalate: fraud, unauthorized charges, billing disputes, account security issues, identity theft, legal threats, account suspensions, any situation where acting without a human risks harming the user
- Escalate if the corpus doesn't contain enough information to answer safely
- Escalate if the ticket is ambiguous and the stakes are high

Reply rules:
- Only reply if the corpus clearly covers the question and the risk is low
- Base your response strictly on the provided corpus excerpts — no outside knowledge, no invented policies
- If the ticket is irrelevant or nonsensical, classify as "invalid" and either reply (out-of-scope notice) or escalate based on risk

Company inference:
- If company is "None", infer from ticket content which product applies
- A ticket may span multiple products — handle the highest-risk thread first"""


def is_high_risk(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in HIGH_RISK_SIGNALS)


def triage(issue: str, subject: str, company: str) -> dict:
    pre_escalate = is_high_risk(issue) or is_high_risk(subject)

    retriever = get_retriever()
    query = f"{subject} {issue}".strip()
    docs = retriever.retrieve(query, company=company, top_k=5)

    if docs:
        corpus_block = "\n\n---\n\n".join(
            f"[{d['source'].upper()} | {d['title']}]\n{d['text']}"
            for d in docs
        )
    else:
        corpus_block = "(no relevant documentation found in corpus)"

    risk_note = (
        "\nNOTE: This ticket contains high-risk signals (fraud/security/billing/access). "
        "You MUST escalate unless the issue is clearly trivial."
    ) if pre_escalate else ""

    user_message = f"""Support ticket:
Company: {company}
Subject: {subject or '(none)'}
Issue: {issue}
{risk_note}

Relevant corpus excerpts:
{corpus_block}

Return the JSON decision now."""

    resp = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=1000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_message}]
    )

    raw = resp.content[0].text.strip()

    # Strip markdown fences if model wrapped output
    if raw.startswith('```'):
        lines = raw.splitlines()
        raw = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])

    result = json.loads(raw)

    # Hard safety override: if we flagged high risk and model said "replied", force escalate
    if pre_escalate and result.get('status') == 'replied':
        result['status'] = 'escalated'
        result['justification'] = (
            "[Safety override — high-risk signal detected] " +
            result.get('justification', '')
        )

    return result